import numpy as np
import tensorflow as tf


class EpisodeEvaluationService:
    """Aggregate episodic metrics over task lists and samplers.

    The service centralizes evaluation-time TensorFlow calls, metric reduction,
    CAN diagnostics, and temporary task-size overrides.
    """

    def __init__(self, *, config, engine, task_pipeline):
        """Initialize the episode evaluation service.

        Args:
            config: Active dataset/training configuration.
            engine: Episodic learning engine used for forward passes.
            task_pipeline: TaskBatchPipeline used for stacking batches.
        """
        self.config = config
        self.engine = engine
        self.task_pipeline = task_pipeline

    @staticmethod
    def split_similarity_scores(
        similarity_scores: np.ndarray, y_true: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split similarities into true-class and other-class groups.

        Args:
            similarity_scores: Query-by-class similarity matrix.
            y_true: True query labels.
        """
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
        """Aggregate intra/inter-class similarity statistics.

        Args:
            intra_class_scores: Similarities assigned to true classes.
            inter_class_scores: Similarities assigned to non-true classes.
        """
        intra_mean = float(np.mean(intra_class_scores))
        inter_mean = float(np.mean(inter_class_scores))
        return {
            "intra_class_similarity": intra_mean,
            "inter_class_similarity": inter_mean,
            "similarity_margin": intra_mean - inter_mean,
        }

    def compute_macro_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute accuracy, macro precision, recall, and F1.

        Args:
            y_true: Ground-truth class labels.
            y_pred: Predicted class labels.
        """
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

    @staticmethod
    def set_sampler_task_size(sampler, k_shot: int, q_query: int) -> None:
        """Update sampler task size in-place for temporary sweeps.

        Args:
            sampler: Episodic sampler with mutable k/q fields.
            k_shot: Support samples per class.
            q_query: Query samples per class.
        """
        sampler.k_shot = int(k_shot)
        sampler.q_query = int(q_query)
        if hasattr(sampler, "n_way"):
            sampler.support_size = int(sampler.n_way * sampler.k_shot)
            sampler.query_size = int(sampler.n_way * sampler.q_query)

    @staticmethod
    def resolve_sampler_task_size(
        sampler,
        k_shot: int,
        q_query: int,
    ) -> tuple[int, int]:
        """Return sampler-aware effective k/q values for temporary sweeps."""
        resolver = getattr(sampler, "resolve_effective_task_size", None)
        if callable(resolver):
            return resolver(k_shot=k_shot, q_query=q_query)
        return int(k_shot), int(q_query)

    def evaluate_task_batch_loss_and_metrics(
        self,
        task_batch: list[dict],
        *,
        forward_batch_size: int | None = None,
        can_support_mode: str | None = None,
    ) -> tuple[float, dict]:
        """Evaluate a task batch and aggregate losses and metrics.

        Args:
            task_batch: List of episodic task dictionaries.
            forward_batch_size: Optional number of tasks per batched forward pass.
            can_support_mode: Optional temporary CAN support mode override.
        """
        if not task_batch:
            raise ValueError("task_batch must contain at least one task")

        original_support_mode = None
        if can_support_mode is not None:
            original_support_mode = self.engine.model.can_support_mode
            self.engine.model.can_support_mode = can_support_mode

        losses = []
        task_losses = []
        can_local_losses = []
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

        try:
            return self._evaluate_task_batch_loss_and_metrics_impl(
                task_batch,
                forward_batch_size=forward_batch_size,
                use_batched_forward=use_batched_forward,
                losses=losses,
                task_losses=task_losses,
                can_local_losses=can_local_losses,
                can_margin_losses=can_margin_losses,
                all_true_tensors=all_true_tensors,
                all_pred_tensors=all_pred_tensors,
                all_intra_class_scores=all_intra_class_scores,
                all_inter_class_scores=all_inter_class_scores,
                all_can_true_scores=all_can_true_scores,
                all_can_best_other_scores=all_can_best_other_scores,
                all_can_score_margins=all_can_score_margins,
            )
        finally:
            if original_support_mode is not None:
                self.engine.model.can_support_mode = original_support_mode

    def _evaluate_task_batch_loss_and_metrics_impl(
        self,
        task_batch: list[dict],
        *,
        forward_batch_size: int | None,
        use_batched_forward: bool,
        losses: list,
        task_losses: list,
        can_local_losses: list,
        can_margin_losses: list,
        all_true_tensors: list,
        all_pred_tensors: list,
        all_intra_class_scores: list,
        all_inter_class_scores: list,
        all_can_true_scores: list,
        all_can_best_other_scores: list,
        all_can_score_margins: list,
    ) -> tuple[float, dict]:
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
                    batch_can_local_losses,
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
                can_local_losses.append(tf.reshape(batch_can_local_losses, [-1]))
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
                can_local_losses.append(
                    tf.reshape(tf.cast(task_outputs["can_local_loss"], tf.float32), [1])
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
        metrics["can_local_loss"] = float(
            tf.reduce_mean(tf.concat(can_local_losses, axis=0))
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
        return float(tf.reduce_mean(tf.concat(losses, axis=0))), metrics

    def evaluate_sampler_loss_and_metrics(
        self,
        sampler,
        num_tasks: int,
        *,
        forward_batch_size: int | None = None,
    ) -> tuple[float, dict]:
        """Sample tasks and evaluate average loss plus metrics.

        Args:
            sampler: Episodic sampler exposing ``get_task``.
            num_tasks: Number of tasks to sample.
            forward_batch_size: Optional number of tasks per batched forward pass.
        """
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
        """Evaluate sampler metrics with temporary k-shot/q-query values.

        Args:
            sampler: Episodic sampler with mutable k/q fields.
            num_tasks: Number of tasks to sample.
            k_shot: Temporary support samples per class.
            q_query: Temporary query samples per class.
            forward_batch_size: Optional batched-forward size.
        """
        original_k = int(sampler.k_shot)
        original_q = int(sampler.q_query)
        effective_k, effective_q = self.resolve_sampler_task_size(
            sampler,
            k_shot=k_shot,
            q_query=q_query,
        )
        self.set_sampler_task_size(sampler, k_shot=effective_k, q_query=effective_q)
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
        """Evaluate one query-only task with prototype memory support.

        Args:
            task_dict: Query-only task dictionary with placeholder support tensors.
        """
        original_support_mode = self.engine.model.can_support_mode
        self.engine.model.can_support_mode = "learned_prototype_memory"
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
            metrics["can_local_loss"] = float(
                tf.reduce_mean(outputs["can_local_losses"])
            )
            metrics["can_margin_loss"] = float(
                tf.reduce_mean(outputs["can_margin_losses"])
            )

            return float(tf.reduce_mean(per_query_loss)), metrics
        finally:
            self.engine.model.can_support_mode = original_support_mode

    def evaluate_source_subject_prototype_vote_task_metrics(
        self,
        *,
        task_dict: dict,
        prototype_maps: np.ndarray,
        prototype_y: np.ndarray,
    ) -> tuple[float, dict, dict[str, np.ndarray]]:
        """Evaluate held-out queries with source-subject prototype voting."""
        if getattr(self.config, "attention_mode", "none") != "can":
            raise ValueError("Source-subject prototype voting requires CAN.")
        softmax_scope = str(
            getattr(
                self.config,
                "source_subject_prototype_vote_softmax_scope",
                "global",
            )
        ).strip().lower()
        if softmax_scope != "global":
            raise ValueError(
                "source_subject_prototype_vote_softmax_scope='global' is "
                "currently the only supported source-subject prototype vote mode."
            )

        query_x = tf.convert_to_tensor(task_dict["query_X"], dtype=tf.float32)
        query_y = tf.convert_to_tensor(task_dict["query_y"], dtype=tf.int32)
        outputs = self.engine.model.forward_source_subject_prototype_vote_can(
            prototype_maps=tf.convert_to_tensor(prototype_maps, dtype=tf.float32),
            prototype_y=tf.convert_to_tensor(prototype_y, dtype=tf.int32),
            query_x=query_x,
            training=False,
        )
        class_probabilities = outputs["similarity_scores"]
        pred = tf.argmax(class_probabilities, axis=1, output_type=tf.int32)
        per_query_loss = tf.keras.losses.sparse_categorical_crossentropy(
            query_y,
            class_probabilities,
            from_logits=False,
        )

        y_true = query_y.numpy().astype(np.int32, copy=False)
        y_pred = pred.numpy().astype(np.int32, copy=False)
        class_scores = class_probabilities.numpy()
        macro = self.compute_macro_metrics(y_true, y_pred)
        intra, inter = self.split_similarity_scores(class_scores, y_true)
        metrics = dict(macro)
        metrics.update(self.compute_similarity_metrics(intra, inter))
        metrics["can_true_class_score"] = float(np.mean(intra))
        best_other_scores = []
        for row_idx, truth in enumerate(y_true):
            row_scores = np.array(class_scores[row_idx], copy=True)
            row_scores[int(truth)] = -np.inf
            best_other_scores.append(float(np.max(row_scores)))
        metrics["can_best_other_score"] = float(np.mean(best_other_scores))
        metrics["can_score_margin"] = (
            metrics["can_true_class_score"] - metrics["can_best_other_score"]
        )
        metrics["can_mean_alignment"] = float(
            tf.reduce_mean(outputs["prototype_similarity_scores"])
        )
        metrics["task_loss"] = float(tf.reduce_mean(per_query_loss))
        metrics["can_local_loss"] = 0.0
        metrics["can_margin_loss"] = 0.0

        diagnostics = {
            "query_y": y_true,
            "query_pred": y_pred,
            "query_loss": per_query_loss.numpy(),
            "query_similarity_scores": class_scores,
            "query_feature_maps": outputs["query_feature_maps"].numpy(),
            "prototype_feature_maps": outputs["prototype_feature_maps"].numpy(),
            "prototype_y": np.asarray(prototype_y, dtype=np.int32),
            "prototype_similarity_scores": outputs[
                "prototype_similarity_scores"
            ].numpy(),
            "prototype_vote_weights": outputs["prototype_vote_weights"].numpy(),
        }
        return float(tf.reduce_mean(per_query_loss)), metrics, diagnostics

    def collect_source_subject_prototype_vote_statistics(
        self,
        *,
        diagnostics: dict[str, np.ndarray],
        phase: str,
    ) -> list[dict]:
        """Collect per-query diagnostics for source-subject prototype voting."""
        rows = []
        query_y = np.asarray(diagnostics["query_y"], dtype=np.int32)
        query_pred = np.asarray(diagnostics["query_pred"], dtype=np.int32)
        losses = np.asarray(diagnostics["query_loss"], dtype=np.float64)
        class_scores = np.asarray(
            diagnostics["query_similarity_scores"], dtype=np.float64
        )
        for sample_index, truth in enumerate(query_y):
            sample_scores = class_scores[sample_index]
            true_score = float(sample_scores[int(truth)])
            other_scores = sample_scores.copy()
            other_scores[int(truth)] = -np.inf
            best_other_score = float(np.max(other_scores))
            row = {
                "phase": phase,
                "task_index": 0,
                "sample_index": sample_index,
                "true_label": int(truth),
                "pred_label": int(query_pred[sample_index]),
                "correct": int(query_pred[sample_index] == int(truth)),
                "loss": float(losses[sample_index]),
                "can_mean_alignment": float(np.mean(sample_scores)),
                "can_true_class_score": true_score,
                "can_best_other_score": best_other_score,
                "can_score_margin": true_score - best_other_score,
            }
            for class_index in range(int(self.config.n_way)):
                row[f"logit_class_{class_index}"] = float(
                    np.log(sample_scores[class_index] + 1e-8)
                )
                row[f"can_score_class_{class_index}"] = float(
                    sample_scores[class_index]
                )
            rows.append(row)
        return rows

    def collect_can_sample_statistics(
        self,
        task_batch: list[dict],
        *,
        phase: str,
        can_support_mode: str | None = None,
    ) -> list[dict]:
        """Collect one diagnostic row per evaluated query sample for CAN.

        Args:
            task_batch: List of episodic task dictionaries.
            phase: Label written into each diagnostic row.
            can_support_mode: Optional temporary CAN support mode override.
        """
        if getattr(self.config, "attention_mode", "none") != "can":
            return []

        original_support_mode = self.engine.model.can_support_mode
        if can_support_mode is not None:
            self.engine.model.can_support_mode = can_support_mode
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
            self.engine.model.can_support_mode = original_support_mode
