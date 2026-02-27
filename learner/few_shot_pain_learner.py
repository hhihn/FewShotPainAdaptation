import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import gc
from data_loaders.pain_meta_dataset import PainMetaDataset
from data_loaders.loso_cross_validator import LOSOCrossValidator
from data_loaders.pain_ds_config import PainDatasetConfig
from utils.logger import setup_logger
from utils.reproducibility import set_global_reproducibility
from utils.training_progress import TrainingProgressReporter
from utils.training_progress_csv import TrainingProgressCSVWriter
from architecture.mulitmodal_proto_net import MultimodalPrototypicalNetwork


class FewShotPainLearner:
    """Meta-learning trainer for personalized pain assessment."""

    def __init__(
        self,
        config: PainDatasetConfig,
        data_dir: str = "./dataset/np-dataset",
        learning_rate: float = 1e-3,
        fusion_method: str = "attention",
        seed: int = 42,
        deterministic_ops: bool = True,
    ):
        """
        Args:
            config: PainDatasetConfig instance
            data_dir: Directory containing numpy files
            learning_rate: Outer loop learning rate
            fusion_method: 'concat', 'mean', or 'attention'
            seed: Global random seed for reproducibility
            deterministic_ops: Enforce deterministic TensorFlow ops where possible
        """
        self.config = config
        self.data_dir = data_dir
        self.learning_rate = learning_rate
        self.fusion_method = fusion_method
        self.seed = seed
        self.deterministic_ops = deterministic_ops
        self.embedding_dim = config.embedding_dim
        self.train_batch_size = max(1, int(config.train_batch_size))
        self.logger = setup_logger("few_shot_pain_learner")
        set_global_reproducibility(
            seed=self.seed,
            deterministic_ops=self.deterministic_ops,
            logger=self.logger,
        )

        # Initialize dataset and cross-validator
        self.dataset = PainMetaDataset(
            data_dir=data_dir, config=config, normalize=True, normalize_per_subject=True
        )

        self.cv = LOSOCrossValidator(
            dataset=self.dataset,
            k_shot=config.k_shot,
            q_query=config.q_query,
            seed=self.seed,
        )

        # Initialize model
        self._rebuild_model(distance_metric="euclidean", clear_session=False)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        run_config = {
            "seed": self.seed,
            "deterministic_ops": self.deterministic_ops,
            "data_dir": self.data_dir,
            "learning_rate": self.learning_rate,
            "fusion_method": self.fusion_method,
            "sequence_length": self.config.sequence_length,
            "n_way": self.config.n_way,
            "k_shot": self.config.k_shot,
            "q_query": self.config.q_query,
            "train_batch_size": self.train_batch_size,
            "embedding_dim": self.embedding_dim,
            "num_tcn_blocks": self.config.num_tcn_blocks,
            "tcn_attention_pool_size": self.config.tcn_attention_pool_size,
            "clear_session_per_fold": self.config.clear_session_per_fold,
            "sensor_idx": list(self.config.sensor_idx),
            "modality_names": list(self.config.modality_names),
        }
        self.logger.info(f"Run config: {json.dumps(run_config, sort_keys=True)}")

        self.logger.info(
            f"Initialized FewShotPainLearner with {len(self.cv.subjects)} subjects"
        )
        num_sensors = len(config.sensor_idx)
        self.logger.info(
            f"Data shape: (sequence_length={config.sequence_length}, num_sensors={num_sensors})"
        )
        self.logger.info(f"Modalities: {config.modality_names}")
        self.logger.info(f"Fusion method: {fusion_method}")

    def _rebuild_model(self, distance_metric: str, clear_session: bool = True) -> None:
        """Build a fresh model/optimizer, optionally clearing stale TF graph state."""
        if clear_session:
            tf.keras.backend.clear_session()
            gc.collect()

        num_sensors = len(self.config.sensor_idx)
        self.model = MultimodalPrototypicalNetwork(
            sequence_length=self.config.sequence_length,
            num_sensors=num_sensors,
            num_classes=self.config.num_stimuli_levels,
            embedding_dim=self.embedding_dim,
            modality_names=self.config.modality_names,
            fusion_method=self.fusion_method,
            distance_metric=distance_metric,
            num_tcn_blocks=self.config.num_tcn_blocks,
            tcn_attention_pool_size=self.config.tcn_attention_pool_size,
        )
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

    def train_step(self, support_x, support_y, query_x, query_y):
        """Single training step on one task."""
        with tf.GradientTape() as tape:
            logits = self.model(support_x, support_y, query_x, training=True)
            loss = self.loss_fn(query_y, logits)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Compute accuracy
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predictions, tf.cast(query_y, tf.int64)), tf.float32)
        )

        return loss, accuracy

    def train_batch_step(self, task_batch: list[dict]) -> tuple[tf.Tensor, tf.Tensor]:
        """Single optimizer update using a batch of tasks."""
        with tf.GradientTape() as tape:
            losses = []
            accuracies = []
            for task_dict in task_batch:
                support_x = tf.constant(task_dict["support_X"], dtype=tf.float32)
                support_y = tf.constant(task_dict["support_y"], dtype=tf.int32)
                query_x = tf.constant(task_dict["query_X"], dtype=tf.float32)
                query_y = tf.constant(task_dict["query_y"], dtype=tf.int32)

                logits = self.model(support_x, support_y, query_x, training=True)
                loss = self.loss_fn(query_y, logits)
                predictions = tf.argmax(logits, axis=1)
                accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(predictions, tf.cast(query_y, tf.int64)), tf.float32)
                )
                losses.append(loss)
                accuracies.append(accuracy)

            batch_loss = tf.reduce_mean(tf.stack(losses))
            batch_acc = tf.reduce_mean(tf.stack(accuracies))

        gradients = tape.gradient(batch_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return batch_loss, batch_acc

    def evaluate_task(self, support_x, support_y, query_x, query_y):
        """Evaluate on one task without updating weights."""
        logits = self.model(support_x, support_y, query_x, training=False)
        loss = self.loss_fn(query_y, logits)

        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predictions, tf.cast(query_y, tf.int64)), tf.float32)
        )

        return loss, accuracy

    def _compute_macro_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute accuracy, macro precision, macro recall, and macro F1."""
        num_classes = self.config.num_stimuli_levels
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

    def _evaluate_sampler_loss_and_metrics(
        self, sampler, num_tasks: int
    ) -> tuple[float, dict]:
        """Evaluate average loss and macro metrics on sampled tasks."""
        losses = []
        all_true = []
        all_pred = []
        for _ in range(num_tasks):
            task_dict = sampler.get_task()

            support_x = tf.constant(task_dict["support_X"], dtype=tf.float32)
            support_y = tf.constant(task_dict["support_y"], dtype=tf.int32)
            query_x = tf.constant(task_dict["query_X"], dtype=tf.float32)
            query_y_np = task_dict["query_y"].astype(np.int32)
            query_y = tf.constant(query_y_np, dtype=tf.int32)

            logits = self.model(support_x, support_y, query_x, training=False)
            loss = self.loss_fn(query_y, logits)
            pred = tf.argmax(logits, axis=1, output_type=tf.int32).numpy()

            losses.append(float(loss))
            all_true.append(query_y_np)
            all_pred.append(pred)

        y_true = np.concatenate(all_true, axis=0)
        y_pred = np.concatenate(all_pred, axis=0)
        return float(np.mean(losses)), self._compute_macro_metrics(y_true, y_pred)

    def train(
        self,
        num_epochs: int = 10,
        tasks_per_epoch: int = 100,
        val_tasks: int = 20,
        training_progress_output_dir: str = "outputs/training_progress",
        k_shot_adaptation_steps: int = 10,
        subject_eval_tasks: int = 20,
        train_log_every: int = 10,
        eval_log_every: int = 5,
    ):
        """
        Train on all subjects using leave-one-subject-out cross-validation.
        """
        cv_results = {
            "train_losses": [],
            "train_accuracies": [],
            "val_losses": [],
            "val_accuracies": [],
            "test_losses": [],
            "test_accuracies": [],
            "zero_shot_losses": [],
            "zero_shot_accuracies": [],
            "zero_shot_precisions": [],
            "zero_shot_recalls": [],
            "zero_shot_f1s": [],
            "k_shot_losses": [],
            "k_shot_accuracies": [],
            "k_shot_precisions": [],
            "k_shot_recalls": [],
            "k_shot_f1s": [],
            "training_progress_files": [],
        }

        num_subjects = len(self.cv.subjects)
        progress = TrainingProgressReporter(
            logger=self.logger,
            train_log_every=train_log_every,
            eval_log_every=eval_log_every,
        )
        csv_writer = TrainingProgressCSVWriter(output_dir=training_progress_output_dir)

        for fold, test_subject in enumerate(self.cv.subjects):
            progress.log_fold_start(
                fold_idx=fold + 1, total_folds=num_subjects, test_subject=test_subject
            )
            progress_file = csv_writer.start_fold(
                fold_idx=fold + 1, test_subject=test_subject
            )

            # Reset model for each fold and free memory from prior graph state.
            self._rebuild_model(
                distance_metric="cosine",
                clear_session=self.config.clear_session_per_fold,
            )

            # Get fold dictionary with samplers
            fold_dict = self.cv.get_fold(test_subject)

            train_sampler = fold_dict["train_sampler"]
            val_sampler = fold_dict["val_sampler"]
            test_sampler = fold_dict["test_sampler"]

            fold_results = {
                "train_losses": [],
                "train_accuracies": [],
                "val_losses": [],
                "val_accuracies": [],
            }

            for epoch in range(num_epochs):
                # Training
                epoch_train_losses = []
                epoch_train_accs = []
                processed_tasks = 0

                for task_start in range(0, tasks_per_epoch, self.train_batch_size):
                    current_batch_size = min(
                        self.train_batch_size, tasks_per_epoch - task_start
                    )
                    task_batch = [
                        train_sampler.get_task() for _ in range(current_batch_size)
                    ]
                    loss, acc = self.train_batch_step(task_batch)
                    processed_tasks += current_batch_size

                    epoch_train_losses.append(float(loss))
                    epoch_train_accs.append(float(acc))
                    csv_writer.write_event(
                        fold_idx=fold + 1,
                        test_subject=test_subject,
                        event_type="train_update",
                        epoch=epoch + 1,
                        epoch_total=num_epochs,
                        step=processed_tasks,
                        step_total=tasks_per_epoch,
                        loss=float(loss),
                        accuracy=float(acc),
                    )
                    progress.log_train_step(
                        fold_idx=fold + 1,
                        total_folds=num_subjects,
                        epoch_idx=epoch + 1,
                        total_epochs=num_epochs,
                        task_idx=processed_tasks,
                        total_tasks=tasks_per_epoch,
                        loss=float(loss),
                        metric_value=float(acc),
                        metric_name="accuracy",
                    )

                # Validation
                epoch_val_losses = []
                epoch_val_accs = []

                for _ in range(val_tasks):
                    task_dict = val_sampler.get_task()

                    support_x = task_dict["support_X"]
                    support_y = task_dict["support_y"]
                    query_x = task_dict["query_X"]
                    query_y = task_dict["query_y"]

                    loss, acc = self.evaluate_task(
                        tf.constant(support_x, dtype=tf.float32),
                        tf.constant(support_y, dtype=tf.int32),
                        tf.constant(query_x, dtype=tf.float32),
                        tf.constant(query_y, dtype=tf.int32),
                    )

                    epoch_val_losses.append(float(loss))
                    epoch_val_accs.append(float(acc))
                    csv_writer.write_event(
                        fold_idx=fold + 1,
                        test_subject=test_subject,
                        event_type="validation_step",
                        epoch=epoch + 1,
                        epoch_total=num_epochs,
                        step=len(epoch_val_losses),
                        step_total=val_tasks,
                        loss=float(loss),
                        accuracy=float(acc),
                    )
                    progress.log_eval_step(
                        stage="Validation",
                        fold_idx=fold + 1,
                        total_folds=num_subjects,
                        step_idx=len(epoch_val_losses),
                        total_steps=val_tasks,
                        loss=float(loss),
                        metric_value=float(acc),
                        metric_name="accuracy",
                    )

                avg_train_loss = np.mean(epoch_train_losses)
                avg_train_acc = np.mean(epoch_train_accs)
                avg_val_loss = np.mean(epoch_val_losses)
                avg_val_acc = np.mean(epoch_val_accs)

                fold_results["train_losses"].append(avg_train_loss)
                fold_results["train_accuracies"].append(avg_train_acc)
                fold_results["val_losses"].append(avg_val_loss)
                fold_results["val_accuracies"].append(avg_val_acc)

                if (epoch + 1) % 2 == 0:
                    self.logger.info(
                        f"Epoch {epoch + 1}/{num_epochs} | "
                        f"Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f} | "
                        f"Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}"
                    )

            # Zero-shot performance (after training on M-1 subjects).
            zero_shot_loss, zero_shot_metrics = self._evaluate_sampler_loss_and_metrics(
                test_sampler, num_tasks=subject_eval_tasks
            )
            csv_writer.write_event(
                fold_idx=fold + 1,
                test_subject=test_subject,
                event_type="zero_shot_summary",
                loss=zero_shot_loss,
                accuracy=zero_shot_metrics["accuracy"],
                precision=zero_shot_metrics["precision"],
                recall=zero_shot_metrics["recall"],
                f1=zero_shot_metrics["f1"],
            )
            progress.log_subject_summary(
                stage="Zero-shot",
                fold_idx=fold + 1,
                total_folds=num_subjects,
                test_subject=int(test_subject),
                loss=zero_shot_loss,
                metrics=zero_shot_metrics,
            )

            # K-shot adaptation on held-out subject data.
            progress.log_adaptation_start(
                fold_idx=fold + 1,
                total_folds=num_subjects,
                test_subject=int(test_subject),
                adaptation_steps=k_shot_adaptation_steps,
            )
            adaptation_losses = []
            for _ in range(k_shot_adaptation_steps):
                adapt_task = test_sampler.get_task()
                support_x = tf.constant(adapt_task["support_X"], dtype=tf.float32)
                support_y = tf.constant(adapt_task["support_y"], dtype=tf.int32)
                query_x = tf.constant(adapt_task["query_X"], dtype=tf.float32)
                query_y = tf.constant(adapt_task["query_y"], dtype=tf.int32)
                adapt_loss, _ = self.train_step(support_x, support_y, query_x, query_y)
                adaptation_losses.append(float(adapt_loss))
            csv_writer.write_event(
                fold_idx=fold + 1,
                test_subject=test_subject,
                event_type="adaptation_phase",
                loss=float(np.mean(adaptation_losses)) if adaptation_losses else 0.0,
            )

            # K-shot performance (after adaptation).
            k_shot_loss, k_shot_metrics = self._evaluate_sampler_loss_and_metrics(
                test_sampler, num_tasks=subject_eval_tasks
            )
            csv_writer.write_event(
                fold_idx=fold + 1,
                test_subject=test_subject,
                event_type="k_shot_summary",
                loss=k_shot_loss,
                accuracy=k_shot_metrics["accuracy"],
                precision=k_shot_metrics["precision"],
                recall=k_shot_metrics["recall"],
                f1=k_shot_metrics["f1"],
            )
            progress.log_subject_summary(
                stage="K-shot",
                fold_idx=fold + 1,
                total_folds=num_subjects,
                test_subject=int(test_subject),
                loss=k_shot_loss,
                metrics=k_shot_metrics,
            )

            cv_results["train_losses"].append(np.mean(fold_results["train_losses"]))
            cv_results["train_accuracies"].append(
                np.mean(fold_results["train_accuracies"])
            )
            cv_results["val_losses"].append(np.mean(fold_results["val_losses"]))
            cv_results["val_accuracies"].append(np.mean(fold_results["val_accuracies"]))
            cv_results["test_losses"].append(zero_shot_loss)
            cv_results["test_accuracies"].append(zero_shot_metrics["accuracy"])
            cv_results["zero_shot_losses"].append(zero_shot_loss)
            cv_results["zero_shot_accuracies"].append(zero_shot_metrics["accuracy"])
            cv_results["zero_shot_precisions"].append(zero_shot_metrics["precision"])
            cv_results["zero_shot_recalls"].append(zero_shot_metrics["recall"])
            cv_results["zero_shot_f1s"].append(zero_shot_metrics["f1"])
            cv_results["k_shot_losses"].append(k_shot_loss)
            cv_results["k_shot_accuracies"].append(k_shot_metrics["accuracy"])
            cv_results["k_shot_precisions"].append(k_shot_metrics["precision"])
            cv_results["k_shot_recalls"].append(k_shot_metrics["recall"])
            cv_results["k_shot_f1s"].append(k_shot_metrics["f1"])
            csv_writer.write_event(
                fold_idx=fold + 1,
                test_subject=test_subject,
                event_type="fold_summary",
                loss=zero_shot_loss,
                accuracy=zero_shot_metrics["accuracy"],
            )
            csv_writer.close()
            cv_results["training_progress_files"].append(progress_file)
            progress.log_fold_complete(
                fold_idx=fold + 1,
                total_folds=num_subjects,
                test_subject=int(test_subject),
                test_loss=zero_shot_loss,
                test_accuracy=zero_shot_metrics["accuracy"],
            )

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("CROSS-VALIDATION RESULTS")
        self.logger.info(f"{'=' * 60}")
        self.logger.info(
            f"Average Zero-shot Accuracy: {np.mean(cv_results['zero_shot_accuracies']):.4f} "
            f"(±{np.std(cv_results['zero_shot_accuracies']):.4f})"
        )
        self.logger.info(
            f"Average K-shot Accuracy: {np.mean(cv_results['k_shot_accuracies']):.4f} "
            f"(±{np.std(cv_results['k_shot_accuracies']):.4f})"
        )
        self.logger.info(
            f"Average Zero-shot Loss: {np.mean(cv_results['zero_shot_losses']):.4f}"
        )
        self.logger.info(
            f"Average K-shot Loss: {np.mean(cv_results['k_shot_losses']):.4f}"
        )
        self.logger.info(f"{'=' * 60}\n")

        return cv_results
