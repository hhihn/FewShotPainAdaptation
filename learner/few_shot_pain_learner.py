import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import gc
import csv
import os
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
        """Single training step on one episode."""
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

    def evaluate_episode(self, support_x, support_y, query_x, query_y):
        """Evaluate on one episode without updating weights."""
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

    def _evaluate_sampler_metrics(self, sampler, num_episodes: int) -> dict:
        """Evaluate classifier metrics on episodes sampled from a sampler."""
        all_true = []
        all_pred = []
        for _ in range(num_episodes):
            episode_dict = sampler.get_episode()

            support_x = tf.constant(episode_dict["support_X"], dtype=tf.float32)
            support_y = tf.constant(episode_dict["support_y"], dtype=tf.int32)
            query_x = tf.constant(episode_dict["query_X"], dtype=tf.float32)
            query_y = episode_dict["query_y"].astype(np.int32)

            logits = self.model(support_x, support_y, query_x, training=False)
            pred = tf.argmax(logits, axis=1, output_type=tf.int32).numpy()

            all_true.append(query_y)
            all_pred.append(pred)

        y_true = np.concatenate(all_true, axis=0)
        y_pred = np.concatenate(all_pred, axis=0)
        return self._compute_macro_metrics(y_true, y_pred)

    def _save_adaptation_curve(
        self,
        records: list,
        output_dir: str,
        fold_index: int,
        test_subject: int,
    ) -> str:
        """Save per-step adaptation metrics to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(
            output_dir,
            f"fold_{fold_index + 1:02d}_subject_{int(test_subject):02d}_adaptation_curve.csv",
        )
        fieldnames = [
            "fold",
            "test_subject",
            "gradient_step",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        return out_path

    def train(
        self,
        num_epochs: int = 10,
        episodes_per_epoch: int = 100,
        val_episodes: int = 20,
        adaptation_steps: int = 30,
        adaptation_eval_episodes: int = 5,
        adaptation_output_dir: str = "outputs/adaptation_curves",
        training_progress_output_dir: str = "outputs/training_progress",
        train_log_every: int = 10,
        eval_log_every: int = 5,
        adaptation_log_every: int = 1,
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
            "adaptation_curve_files": [],
            "training_progress_files": [],
        }

        num_subjects = len(self.cv.subjects)
        progress = TrainingProgressReporter(
            logger=self.logger,
            train_log_every=train_log_every,
            eval_log_every=eval_log_every,
            adaptation_log_every=adaptation_log_every,
        )
        csv_writer = TrainingProgressCSVWriter(output_dir=training_progress_output_dir)

        for fold, test_subject in enumerate(self.cv.subjects):
            progress.log_fold_start(
                fold_idx=fold + 1, total_folds=num_subjects, test_subject=test_subject
            )
            progress_file = csv_writer.start_fold(fold_idx=fold + 1, test_subject=test_subject)

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

                for episode in range(episodes_per_epoch):
                    episode_dict = train_sampler.get_episode()

                    support_x = episode_dict["support_X"]  # [6*k_shot, 2500, 3]
                    support_y = episode_dict["support_y"]  # [6*k_shot]
                    query_x = episode_dict["query_X"]  # [6*q_query, 2500, 3]
                    query_y = episode_dict["query_y"]  # [6*q_query]

                    loss, acc = self.train_step(
                        tf.constant(support_x, dtype=tf.float32),
                        tf.constant(support_y, dtype=tf.int32),
                        tf.constant(query_x, dtype=tf.float32),
                        tf.constant(query_y, dtype=tf.int32),
                    )

                    epoch_train_losses.append(float(loss))
                    epoch_train_accs.append(float(acc))
                    csv_writer.write_event(
                        fold_idx=fold + 1,
                        test_subject=test_subject,
                        event_type="train_step",
                        epoch=epoch + 1,
                        epoch_total=num_epochs,
                        step=episode + 1,
                        step_total=episodes_per_epoch,
                        loss=float(loss),
                        accuracy=float(acc),
                    )
                    progress.log_train_step(
                        fold_idx=fold + 1,
                        total_folds=num_subjects,
                        epoch_idx=epoch + 1,
                        total_epochs=num_epochs,
                        episode_idx=episode + 1,
                        total_episodes=episodes_per_epoch,
                        loss=float(loss),
                        metric_value=float(acc),
                        metric_name="accuracy",
                    )

                # Validation
                epoch_val_losses = []
                epoch_val_accs = []

                for _ in range(val_episodes):
                    episode_dict = val_sampler.get_episode()

                    support_x = episode_dict["support_X"]
                    support_y = episode_dict["support_y"]
                    query_x = episode_dict["query_X"]
                    query_y = episode_dict["query_y"]

                    loss, acc = self.evaluate_episode(
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
                        step_total=val_episodes,
                        loss=float(loss),
                        accuracy=float(acc),
                    )
                    progress.log_eval_step(
                        stage="Validation",
                        fold_idx=fold + 1,
                        total_folds=num_subjects,
                        step_idx=len(epoch_val_losses),
                        total_steps=val_episodes,
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

            # Test on held-out subject
            test_losses = []
            test_accs = []

            for _ in range(val_episodes):
                episode_dict = test_sampler.get_episode()

                support_x = episode_dict["support_X"]
                support_y = episode_dict["support_y"]
                query_x = episode_dict["query_X"]
                query_y = episode_dict["query_y"]

                loss, acc = self.evaluate_episode(
                    tf.constant(support_x, dtype=tf.float32),
                    tf.constant(support_y, dtype=tf.int32),
                    tf.constant(query_x, dtype=tf.float32),
                    tf.constant(query_y, dtype=tf.int32),
                )

                test_losses.append(float(loss))
                test_accs.append(float(acc))
                csv_writer.write_event(
                    fold_idx=fold + 1,
                    test_subject=test_subject,
                    event_type="test_step",
                    epoch=None,
                    epoch_total=None,
                    step=len(test_losses),
                    step_total=val_episodes,
                    loss=float(loss),
                    accuracy=float(acc),
                )
                progress.log_eval_step(
                    stage="Test",
                    fold_idx=fold + 1,
                    total_folds=num_subjects,
                    step_idx=len(test_losses),
                    total_steps=val_episodes,
                    loss=float(loss),
                    metric_value=float(acc),
                    metric_name="accuracy",
                )

            avg_test_loss = np.mean(test_losses)
            avg_test_acc = np.mean(test_accs)

            self.logger.info(
                f"\nTest Subject {test_subject}: "
                f"Loss: {avg_test_loss:.4f}, Accuracy: {avg_test_acc:.4f}"
            )

            # Adaptation-speed evaluation on the held-out subject.
            adaptation_records = []
            for step_idx in range(1, adaptation_steps + 1):
                adapt_episode = test_sampler.get_episode()
                support_x = tf.constant(adapt_episode["support_X"], dtype=tf.float32)
                support_y = tf.constant(adapt_episode["support_y"], dtype=tf.int32)
                query_x = tf.constant(adapt_episode["query_X"], dtype=tf.float32)
                query_y = tf.constant(adapt_episode["query_y"], dtype=tf.int32)
                adapt_loss, _ = self.train_step(support_x, support_y, query_x, query_y)

                step_metrics = self._evaluate_sampler_metrics(
                    test_sampler, num_episodes=adaptation_eval_episodes
                )
                csv_writer.write_event(
                    fold_idx=fold + 1,
                    test_subject=test_subject,
                    event_type="adaptation_step",
                    epoch=None,
                    epoch_total=None,
                    step=step_idx,
                    step_total=adaptation_steps,
                    loss=float(adapt_loss),
                    accuracy=step_metrics["accuracy"],
                    precision=step_metrics["precision"],
                    recall=step_metrics["recall"],
                    f1=step_metrics["f1"],
                )
                progress.log_adaptation_step(
                    fold_idx=fold + 1,
                    total_folds=num_subjects,
                    test_subject=int(test_subject),
                    step_idx=step_idx,
                    total_steps=adaptation_steps,
                    loss=float(adapt_loss),
                    metrics=step_metrics,
                )
                adaptation_records.append(
                    {
                        "fold": fold + 1,
                        "test_subject": int(test_subject),
                        "gradient_step": step_idx,
                        "accuracy": step_metrics["accuracy"],
                        "precision": step_metrics["precision"],
                        "recall": step_metrics["recall"],
                        "f1": step_metrics["f1"],
                    }
                )

            curve_file = self._save_adaptation_curve(
                records=adaptation_records,
                output_dir=adaptation_output_dir,
                fold_index=fold,
                test_subject=test_subject,
            )
            cv_results["adaptation_curve_files"].append(curve_file)
            self.logger.info(
                f"Saved adaptation curve for subject {test_subject} to {curve_file}"
            )

            cv_results["train_losses"].append(np.mean(fold_results["train_losses"]))
            cv_results["train_accuracies"].append(
                np.mean(fold_results["train_accuracies"])
            )
            cv_results["val_losses"].append(np.mean(fold_results["val_losses"]))
            cv_results["val_accuracies"].append(np.mean(fold_results["val_accuracies"]))
            cv_results["test_losses"].append(avg_test_loss)
            cv_results["test_accuracies"].append(avg_test_acc)
            csv_writer.write_event(
                fold_idx=fold + 1,
                test_subject=test_subject,
                event_type="fold_summary",
                loss=float(avg_test_loss),
                accuracy=float(avg_test_acc),
            )
            csv_writer.close()
            cv_results["training_progress_files"].append(progress_file)
            progress.log_fold_complete(
                fold_idx=fold + 1,
                total_folds=num_subjects,
                test_subject=int(test_subject),
                test_loss=avg_test_loss,
                test_accuracy=avg_test_acc,
            )

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("CROSS-VALIDATION RESULTS")
        self.logger.info(f"{'=' * 60}")
        self.logger.info(
            f"Average Test Accuracy: {np.mean(cv_results['test_accuracies']):.4f} "
            f"(±{np.std(cv_results['test_accuracies']):.4f})"
        )
        self.logger.info(f"Average Test Loss: {np.mean(cv_results['test_losses']):.4f}")
        self.logger.info(f"{'=' * 60}\n")

        return cv_results
