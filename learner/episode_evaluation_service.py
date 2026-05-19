import numpy as np
import tensorflow as tf


class EpisodeEvaluationService:
    """Aggregate episodic task metrics over task lists and samplers."""

    def __init__(self, *, config, engine, task_pipeline):
        self.config = config
        self.engine = engine
        self.task_pipeline = task_pipeline

    @staticmethod
    def split_similarity_scores(
        similarity_scores: np.ndarray, y_true: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split query-to-prototype similarities into intra/inter-class groups."""
        row_indices = np.arange(len(y_true))
        intra_class_scores = similarity_scores[row_indices, y_true]

        inter_class_mask = np.ones_like(similarity_scores, dtype=bool)
        inter_class_mask[row_indices, y_true] = False
        inter_class_scores = similarity_scores[inter_class_mask]

        return intra_class_scores, inter_class_scores

    @staticmethod
    def compute_similarity_metrics(
        intra_class_scores: np.ndarray, inter_class_scores: np.ndarray
    ) -> dict:
        """Aggregate similarity statistics using the existing metric dict shape."""
        intra_mean = float(np.mean(intra_class_scores))
        inter_mean = float(np.mean(inter_class_scores))
        return {
            "intra_class_similarity": intra_mean,
            "inter_class_similarity": inter_mean,
            "similarity_margin": intra_mean - inter_mean,
        }

    def compute_macro_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute accuracy, macro precision, macro recall, and macro F1."""
        num_classes = self.config.n_way
        conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)
        for truth, pred in zip(y_true, y_pred):
            conf_mat[int(truth), int(pred)] += 1

        tp = np.diag(conf_mat).astype(np.float64)
        fp = np.sum(conf_mat, axis=0) - tp
        fn = np.sum(conf_mat, axis=1) - tp

        precision_per_class = np.divide(
            tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0
        )
        recall_per_class = np.divide(
            tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0
        )
        f1_per_class = np.divide(
            2 * precision_per_class * recall_per_class,
            precision_per_class + recall_per_class,
            out=np.zeros_like(tp),
            where=(precision_per_class + recall_per_class) > 0,
        )

        total = np.sum(conf_mat)
        accuracy = float(np.sum(tp) / total) if total > 0 else 0.0
        return {
            "accuracy": accuracy,
            "precision": float(np.mean(precision_per_class)),
            "recall": float(np.mean(recall_per_class)),
            "f1": float(np.mean(f1_per_class)),
        }

    def evaluate_transductive_task_batch_metrics(self, task_batch: list[dict]) -> dict:
        """Evaluate CAN transductive predictions without changing inductive metrics."""
        if getattr(self.config, "attention_mode", "none") != "can":
            return {}
        if int(getattr(self.config, "can_transductive_iterations", 0)) <= 0:
            return {}

        losses = []
        all_true = []
        all_pred = []
        all_intra_scores = []
        all_inter_scores = []
        class_ids = tf.range(int(self.config.n_way), dtype=tf.int32)[tf.newaxis, :]
        for task_dict in task_batch:
            support_x = tf.convert_to_tensor(task_dict["support_X"], dtype=tf.float32)
            support_y = tf.convert_to_tensor(task_dict["support_y"], dtype=tf.int32)
            query_x = tf.convert_to_tensor(task_dict["query_X"], dtype=tf.float32)
            query_y = tf.convert_to_tensor(task_dict["query_y"], dtype=tf.int32)
            outputs = self.engine.model.forward_episode_batch_transductive(
                support_x=support_x[tf.newaxis, ...],
                support_y=support_y[tf.newaxis, ...],
                query_x=query_x[tf.newaxis, ...],
                training=False,
            )
            logits = outputs["transductive_logits"][0]
            similarity_scores = outputs["transductive_similarity_scores"][0]
            pred = tf.argmax(logits, axis=1, output_type=tf.int32)
            losses.append(
                tf.reshape(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        query_y,
                        logits,
                        from_logits=True,
                    ),
                    [-1],
                )
            )
            row_indices = tf.range(tf.shape(query_y)[0], dtype=tf.int32)
            intra_class_scores = tf.gather_nd(
                similarity_scores,
                tf.stack([row_indices, query_y], axis=1),
            )
            inter_class_mask = tf.not_equal(class_ids, query_y[:, tf.newaxis])
            all_intra_scores.append(tf.reshape(intra_class_scores, [-1]))
            all_inter_scores.append(
                tf.reshape(tf.boolean_mask(similarity_scores, inter_class_mask), [-1])
            )
            all_true.append(tf.reshape(query_y, [-1]))
            all_pred.append(tf.reshape(pred, [-1]))

        y_true = tf.concat(all_true, axis=0).numpy().astype(np.int32, copy=False)
        y_pred = tf.concat(all_pred, axis=0).numpy().astype(np.int32, copy=False)
        macro = self.compute_macro_metrics(y_true, y_pred)
        similarity = self.compute_similarity_metrics(
            tf.concat(all_intra_scores, axis=0).numpy(),
            tf.concat(all_inter_scores, axis=0).numpy(),
        )
        return {
            "transductive_loss": float(tf.reduce_mean(tf.concat(losses, axis=0))),
            "transductive_accuracy": macro["accuracy"],
            "transductive_precision": macro["precision"],
            "transductive_recall": macro["recall"],
            "transductive_f1": macro["f1"],
            "transductive_intra_class_similarity": similarity["intra_class_similarity"],
            "transductive_inter_class_similarity": similarity["inter_class_similarity"],
            "transductive_similarity_margin": similarity["similarity_margin"],
        }

    @staticmethod
    def set_sampler_task_size(sampler, k_shot: int, q_query: int) -> None:
        """Update sampler task size in-place for temporary held-out evaluation sweeps."""
        sampler.k_shot = int(k_shot)
        sampler.q_query = int(q_query)
        if hasattr(sampler, "n_way"):
            sampler.support_size = int(sampler.n_way * sampler.k_shot)
            sampler.query_size = int(sampler.n_way * sampler.q_query)

    def evaluate_task_batch_loss_and_metrics(
        self,
        task_batch: list[dict],
        *,
        forward_batch_size: int | None = None,
    ) -> tuple[float, dict]:
        """Evaluate a task batch and aggregate classification/similarity metrics."""
        if not task_batch:
            raise ValueError("task_batch must contain at least one task")

        losses = []
        task_losses = []
        contrastive_losses = []
        triplet_losses = []
        can_local_losses = []
        can_global_losses = []
        can_margin_losses = []
        all_true_tensors = []
        all_pred_tensors = []
        all_intra_class_scores = []
        all_inter_class_scores = []
        all_can_true_scores = []
        all_can_best_other_scores = []
        all_can_score_margins = []
        use_batched_forward = (
            forward_batch_size is not None and int(forward_batch_size) > 1
        )

        if use_batched_forward and self.task_pipeline.task_batch_has_uniform_shapes(
            task_batch
        ):
            (
                support_x_batch,
                support_y_batch,
                query_x_batch,
                query_y_batch,
            ) = self.task_pipeline.stack_task_batch(task_batch)
            eval_batch_size = max(1, int(forward_batch_size))
            total_tasks = len(task_batch)
            for task_start in range(0, total_tasks, eval_batch_size):
                task_end = min(total_tasks, task_start + eval_batch_size)
                (
                    batch_losses,
                    batch_task_losses,
                    batch_contrastive_losses,
                    batch_triplet_losses,
                    batch_can_local_losses,
                    batch_can_global_losses,
                    batch_can_margin_losses,
                    batch_y_true,
                    batch_y_pred,
                    batch_intra_scores,
                    batch_inter_scores,
                    batch_can_true_scores,
                    batch_can_best_other_scores,
                    batch_can_score_margins,
                ) = self.engine.eval_task_batch_step_tensors(
                    support_x_batch=support_x_batch[task_start:task_end],
                    support_y_batch=support_y_batch[task_start:task_end],
                    query_x_batch=query_x_batch[task_start:task_end],
                    query_y_batch=query_y_batch[task_start:task_end],
                )
                losses.append(tf.reshape(batch_losses, [-1]))
                task_losses.append(tf.reshape(batch_task_losses, [-1]))
                contrastive_losses.append(tf.reshape(batch_contrastive_losses, [-1]))
                triplet_losses.append(tf.reshape(batch_triplet_losses, [-1]))
                can_local_losses.append(tf.reshape(batch_can_local_losses, [-1]))
                can_global_losses.append(tf.reshape(batch_can_global_losses, [-1]))
                can_margin_losses.append(tf.reshape(batch_can_margin_losses, [-1]))
                all_true_tensors.append(tf.reshape(batch_y_true, [-1]))
                all_pred_tensors.append(tf.reshape(batch_y_pred, [-1]))
                all_intra_class_scores.append(tf.reshape(batch_intra_scores, [-1]))
                all_inter_class_scores.append(tf.reshape(batch_inter_scores, [-1]))
                all_can_true_scores.append(tf.reshape(batch_can_true_scores, [-1]))
                all_can_best_other_scores.append(
                    tf.reshape(batch_can_best_other_scores, [-1])
                )
                all_can_score_margins.append(tf.reshape(batch_can_score_margins, [-1]))
        else:
            class_ids = tf.range(int(self.config.n_way), dtype=tf.int32)[tf.newaxis, :]
            if self.task_pipeline.task_batch_has_uniform_shapes(task_batch):
                (
                    support_x_batch,
                    support_y_batch,
                    query_x_batch,
                    query_y_batch,
                ) = self.task_pipeline.stack_task_batch(task_batch)
                task_iter = zip(
                    tf.unstack(support_x_batch, axis=0),
                    tf.unstack(support_y_batch, axis=0),
                    tf.unstack(query_x_batch, axis=0),
                    tf.unstack(query_y_batch, axis=0),
                )
            else:
                task_iter = (
                    (
                        tf.convert_to_tensor(task_dict["support_X"], dtype=tf.float32),
                        tf.convert_to_tensor(task_dict["support_y"], dtype=tf.int32),
                        tf.convert_to_tensor(task_dict["query_X"], dtype=tf.float32),
                        tf.convert_to_tensor(task_dict["query_y"], dtype=tf.int32),
                    )
                    for task_dict in task_batch
                )

            for support_x, support_y, query_x, query_y in task_iter:
                task_outputs = self.engine.forward_task(
                    support_x,
                    support_y,
                    query_x,
                    query_y,
                    training=False,
                    return_similarity_scores=True,
                )
                logits = task_outputs["logits"]
                similarity_scores = task_outputs["similarity_scores"]
                pred = tf.argmax(logits, axis=1, output_type=tf.int32)
                row_indices = tf.range(tf.shape(query_y)[0], dtype=tf.int32)
                intra_class_scores = tf.gather_nd(
                    similarity_scores,
                    tf.stack([row_indices, query_y], axis=1),
                )
                inter_class_mask = tf.not_equal(class_ids, query_y[:, tf.newaxis])
                inter_class_scores = tf.boolean_mask(
                    similarity_scores, inter_class_mask
                )
                best_other_scores = tf.reduce_max(
                    tf.where(
                        inter_class_mask,
                        similarity_scores,
                        tf.fill(
                            tf.shape(similarity_scores),
                            tf.cast(-1e9, similarity_scores.dtype),
                        ),
                    ),
                    axis=1,
                )
                can_score_margins = intra_class_scores - best_other_scores

                losses.append(
                    tf.reshape(tf.cast(task_outputs["loss"], tf.float32), [1])
                )
                task_losses.append(
                    tf.reshape(tf.cast(task_outputs["task_loss"], tf.float32), [1])
                )
                contrastive_losses.append(
                    tf.reshape(
                        tf.cast(task_outputs["contrastive_loss"], tf.float32), [1]
                    )
                )
                triplet_losses.append(
                    tf.reshape(tf.cast(task_outputs["triplet_loss"], tf.float32), [1])
                )
                can_local_losses.append(
                    tf.reshape(tf.cast(task_outputs["can_local_loss"], tf.float32), [1])
                )
                can_global_losses.append(
                    tf.reshape(
                        tf.cast(task_outputs["can_global_loss"], tf.float32), [1]
                    )
                )
                can_margin_losses.append(
                    tf.reshape(
                        tf.cast(task_outputs["can_margin_loss"], tf.float32), [1]
                    )
                )
                all_true_tensors.append(tf.reshape(query_y, [-1]))
                all_pred_tensors.append(tf.reshape(pred, [-1]))
                all_intra_class_scores.append(tf.reshape(intra_class_scores, [-1]))
                all_inter_class_scores.append(tf.reshape(inter_class_scores, [-1]))
                all_can_true_scores.append(tf.reshape(intra_class_scores, [-1]))
                all_can_best_other_scores.append(tf.reshape(best_other_scores, [-1]))
                all_can_score_margins.append(tf.reshape(can_score_margins, [-1]))

        y_true = (
            tf.concat(all_true_tensors, axis=0).numpy().astype(np.int32, copy=False)
        )
        y_pred = (
            tf.concat(all_pred_tensors, axis=0).numpy().astype(np.int32, copy=False)
        )
        metrics = self.compute_macro_metrics(y_true, y_pred)
        intra_scores = tf.concat(all_intra_class_scores, axis=0)
        inter_scores = tf.concat(all_inter_class_scores, axis=0)
        metrics.update(
            self.compute_similarity_metrics(
                intra_scores.numpy(),
                inter_scores.numpy(),
            )
        )
        metrics["task_loss"] = float(tf.reduce_mean(tf.concat(task_losses, axis=0)))
        metrics["contrastive_loss"] = float(
            tf.reduce_mean(tf.concat(contrastive_losses, axis=0))
        )
        metrics["triplet_loss"] = float(
            tf.reduce_mean(tf.concat(triplet_losses, axis=0))
        )
        metrics["can_local_loss"] = float(
            tf.reduce_mean(tf.concat(can_local_losses, axis=0))
        )
        metrics["can_global_loss"] = float(
            tf.reduce_mean(tf.concat(can_global_losses, axis=0))
        )
        metrics["can_margin_loss"] = float(
            tf.reduce_mean(tf.concat(can_margin_losses, axis=0))
        )
        if getattr(self.config, "attention_mode", "none") == "can":
            metrics["can_true_class_score"] = float(
                tf.reduce_mean(tf.concat(all_can_true_scores, axis=0))
            )
            metrics["can_best_other_score"] = float(
                tf.reduce_mean(tf.concat(all_can_best_other_scores, axis=0))
            )
            metrics["can_score_margin"] = float(
                tf.reduce_mean(tf.concat(all_can_score_margins, axis=0))
            )
            metrics["can_mean_alignment"] = float(
                tf.reduce_mean(tf.concat([intra_scores, inter_scores], axis=0))
            )
        metrics.update(self.evaluate_transductive_task_batch_metrics(task_batch))
        return float(tf.reduce_mean(tf.concat(losses, axis=0))), metrics

    def evaluate_sampler_loss_and_metrics(
        self,
        sampler,
        num_tasks: int,
        *,
        forward_batch_size: int | None = None,
    ) -> tuple[float, dict]:
        """Evaluate average loss plus classification/similarity metrics on tasks."""
        return self.evaluate_task_batch_loss_and_metrics(
            [sampler.get_task() for _ in range(num_tasks)],
            forward_batch_size=forward_batch_size,
        )

    def evaluate_sampler_loss_and_metrics_at_task_size(
        self,
        sampler,
        num_tasks: int,
        *,
        k_shot: int,
        q_query: int,
        forward_batch_size: int | None = None,
    ) -> tuple[float, dict]:
        """Evaluate sampler metrics with a temporary k-shot/q-query override."""
        original_k = int(sampler.k_shot)
        original_q = int(sampler.q_query)
        self.set_sampler_task_size(sampler, k_shot=k_shot, q_query=q_query)
        try:
            return self.evaluate_sampler_loss_and_metrics(
                sampler,
                num_tasks=num_tasks,
                forward_batch_size=forward_batch_size,
            )
        finally:
            self.set_sampler_task_size(sampler, k_shot=original_k, q_query=original_q)

    def evaluate_prototype_memory_task_metrics(
        self, task_dict: dict
    ) -> tuple[float, dict]:
        """Evaluate one query-only task with learned prototype memory as support."""
        original_support_mode = self.engine.model.can_support_mode
        original_triplet_weight = self.engine.triplet_loss_weight
        self.engine.model.can_support_mode = "learned_prototype_memory"
        self.engine.triplet_loss_weight = 0.0
        try:
            support_x = tf.convert_to_tensor(task_dict["support_X"], dtype=tf.float32)[
                tf.newaxis, ...
            ]
            support_y = tf.convert_to_tensor(task_dict["support_y"], dtype=tf.int32)[
                tf.newaxis, ...
            ]
            query_x = tf.convert_to_tensor(task_dict["query_X"], dtype=tf.float32)[
                tf.newaxis, ...
            ]
            query_y = tf.convert_to_tensor(task_dict["query_y"], dtype=tf.int32)[
                tf.newaxis, ...
            ]
            outputs = self.engine.forward_task_batch(
                support_x_batch=support_x,
                support_y_batch=support_y,
                query_x_batch=query_x,
                query_y_batch=query_y,
                training=False,
                return_similarity_scores=True,
            )
            logits = outputs["logits"][0]
            similarity_scores = outputs["similarity_scores"][0]
            query_y_flat = query_y[0]
            pred = tf.argmax(logits, axis=1, output_type=tf.int32)
            per_query_loss = tf.keras.losses.sparse_categorical_crossentropy(
                query_y_flat,
                logits,
                from_logits=True,
            )

            y_true = query_y_flat.numpy().astype(np.int32, copy=False)
            y_pred = pred.numpy().astype(np.int32, copy=False)
            macro = self.compute_macro_metrics(y_true, y_pred)
            intra, inter = self.split_similarity_scores(
                similarity_scores.numpy(),
                y_true,
            )
            metrics = dict(macro)
            metrics.update(self.compute_similarity_metrics(intra, inter))
            metrics["can_true_class_score"] = float(np.mean(intra))
            best_other_scores = []
            for row_idx, truth in enumerate(y_true):
                row_scores = np.array(similarity_scores[row_idx].numpy(), copy=True)
                row_scores[int(truth)] = -np.inf
                best_other_scores.append(float(np.max(row_scores)))
            metrics["can_best_other_score"] = float(np.mean(best_other_scores))
            metrics["can_score_margin"] = (
                metrics["can_true_class_score"] - metrics["can_best_other_score"]
            )
            metrics["can_mean_alignment"] = float(tf.reduce_mean(similarity_scores))
            metrics["task_loss"] = float(tf.reduce_mean(per_query_loss))
            metrics["contrastive_loss"] = 0.0
            metrics["triplet_loss"] = 0.0
            metrics["can_local_loss"] = float(
                tf.reduce_mean(outputs["can_local_losses"])
            )
            metrics["can_global_loss"] = float(
                tf.reduce_mean(outputs["can_global_losses"])
            )
            metrics["can_margin_loss"] = float(
                tf.reduce_mean(outputs["can_margin_losses"])
            )

            if (
                getattr(self.config, "attention_mode", "none") == "can"
                and int(getattr(self.config, "can_transductive_iterations", 0)) > 0
            ):
                transductive = self.engine.model.forward_episode_batch_transductive(
                    support_x=support_x,
                    support_y=support_y,
                    query_x=query_x,
                    training=False,
                )
                transductive_logits = transductive["transductive_logits"][0]
                transductive_scores = transductive["transductive_similarity_scores"][0]
                transductive_pred = tf.argmax(
                    transductive_logits,
                    axis=1,
                    output_type=tf.int32,
                )
                transductive_macro = self.compute_macro_metrics(
                    y_true,
                    transductive_pred.numpy().astype(np.int32, copy=False),
                )
                trans_intra, trans_inter = self.split_similarity_scores(
                    transductive_scores.numpy(),
                    y_true,
                )
                trans_similarity = self.compute_similarity_metrics(
                    trans_intra,
                    trans_inter,
                )
                metrics.update(
                    {
                        "transductive_loss": float(
                            tf.reduce_mean(
                                tf.keras.losses.sparse_categorical_crossentropy(
                                    query_y_flat,
                                    transductive_logits,
                                    from_logits=True,
                                )
                            )
                        ),
                        "transductive_accuracy": transductive_macro["accuracy"],
                        "transductive_precision": transductive_macro["precision"],
                        "transductive_recall": transductive_macro["recall"],
                        "transductive_f1": transductive_macro["f1"],
                        "transductive_intra_class_similarity": trans_similarity[
                            "intra_class_similarity"
                        ],
                        "transductive_inter_class_similarity": trans_similarity[
                            "inter_class_similarity"
                        ],
                        "transductive_similarity_margin": trans_similarity[
                            "similarity_margin"
                        ],
                        "transductive_selected_count": int(
                            tf.reduce_sum(
                                tf.cast(
                                    transductive["transductive_selected_mask"],
                                    tf.int32,
                                )
                            )
                        ),
                    }
                )
            return float(tf.reduce_mean(per_query_loss)), metrics
        finally:
            self.engine.triplet_loss_weight = original_triplet_weight
            self.engine.model.can_support_mode = original_support_mode

    def collect_can_sample_statistics(
        self,
        task_batch: list[dict],
        *,
        phase: str,
        can_support_mode: str | None = None,
    ) -> list[dict]:
        """Return one diagnostic row per evaluated query sample for CAN folds."""
        if getattr(self.config, "attention_mode", "none") != "can":
            return []

        original_support_mode = self.engine.model.can_support_mode
        original_triplet_weight = self.engine.triplet_loss_weight
        if can_support_mode is not None:
            self.engine.model.can_support_mode = can_support_mode
        self.engine.triplet_loss_weight = 0.0
        try:
            rows = []
            for task_index, task_dict in enumerate(task_batch):
                support_x = tf.convert_to_tensor(
                    task_dict["support_X"],
                    dtype=tf.float32,
                )
                support_y = tf.convert_to_tensor(
                    task_dict["support_y"],
                    dtype=tf.int32,
                )
                query_x = tf.convert_to_tensor(
                    task_dict["query_X"],
                    dtype=tf.float32,
                )
                query_y = tf.convert_to_tensor(
                    task_dict["query_y"],
                    dtype=tf.int32,
                )
                outputs = self.engine.forward_task(
                    support_x=support_x,
                    support_y=support_y,
                    query_x=query_x,
                    query_y=query_y,
                    training=False,
                    return_similarity_scores=True,
                )
                logits = outputs["logits"]
                similarity_scores = outputs["similarity_scores"]
                pred = tf.argmax(logits, axis=1, output_type=tf.int32)
                losses = tf.keras.losses.sparse_categorical_crossentropy(
                    query_y,
                    logits,
                    from_logits=True,
                )

                query_y_np = query_y.numpy().astype(np.int32, copy=False)
                pred_np = pred.numpy().astype(np.int32, copy=False)
                losses_np = losses.numpy()
                logits_np = logits.numpy()
                scores_np = similarity_scores.numpy()

                for sample_index, truth in enumerate(query_y_np):
                    sample_scores = np.asarray(
                        scores_np[sample_index], dtype=np.float64
                    )
                    true_score = float(sample_scores[int(truth)])
                    other_scores = sample_scores.copy()
                    other_scores[int(truth)] = -np.inf
                    best_other_score = float(np.max(other_scores))
                    row = {
                        "phase": phase,
                        "task_index": task_index,
                        "sample_index": sample_index,
                        "true_label": int(truth),
                        "pred_label": int(pred_np[sample_index]),
                        "correct": int(pred_np[sample_index] == int(truth)),
                        "loss": float(losses_np[sample_index]),
                        "can_mean_alignment": float(np.mean(sample_scores)),
                        "can_true_class_score": true_score,
                        "can_best_other_score": best_other_score,
                        "can_score_margin": true_score - best_other_score,
                    }
                    for class_index in range(int(self.config.n_way)):
                        row[f"logit_class_{class_index}"] = float(
                            logits_np[sample_index, class_index]
                        )
                        row[f"can_score_class_{class_index}"] = float(
                            scores_np[sample_index, class_index]
                        )
                    rows.append(row)
            return rows
        finally:
            self.engine.triplet_loss_weight = original_triplet_weight
            self.engine.model.can_support_mode = original_support_mode
