import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import gc
import io
import os
import time
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
        fusion_method: str = "mean",
        distance_metric: str = "cosine",
    ):
        """
        Args:
            config: PainDatasetConfig instance
            data_dir: Directory containing numpy files
            learning_rate: Outer loop learning rate
            fusion_method: 'mean', 'gated', or 'transformer_ib'
        """
        self.config = config
        self.data_dir = data_dir
        self.learning_rate = learning_rate
        self.fusion_method = fusion_method
        self.distance_metric = distance_metric
        self.seed = int(config.seed)
        self.deterministic_ops = bool(config.deterministic_ops)
        self.embedding_dim = config.embedding_dim
        self.train_batch_size = max(1, int(config.train_batch_size))
        self.supcon_loss_weight = float(config.supcon_loss_weight)
        self.supcon_temperature = float(config.supcon_temperature)
        self.triplet_loss_weight = float(config.triplet_loss_weight)
        self.triplet_margin = float(config.triplet_margin)
        self.gaussian_noise_std = float(config.gaussian_noise_std)
        self.support_size = int(self.config.n_way * self.config.k_shot)
        self.query_size = int(self.config.n_way * self.config.q_query)
        self.num_sensors = int(len(self.config.sensor_idx))
        self.sequence_length = int(self.config.sequence_length)
        self.train_prefetch_batches = max(
            1, int(getattr(self.config, "train_prefetch_batches", 2))
        )
        self._compiled_train_batch_step = None
        self.logging_verbosity = int(getattr(config, "logging_verbosity", 1))
        self.logger = setup_logger("few_shot_pain_learner")
        if self.logging_verbosity <= 0:
            self.logger.setLevel(30)
        elif self.logging_verbosity == 1:
            self.logger.setLevel(20)
        else:
            self.logger.setLevel(10)
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
            seed=self.seed,
        )

        # Initialize model
        self._rebuild_model(clear_session=False)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        run_config = {
            "seed": self.seed,
            "deterministic_ops": self.deterministic_ops,
            "data_dir": self.data_dir,
            "dataset_source": self.config.dataset_source,
            "split_strategy": self.config.split_strategy,
            "learning_rate": self.learning_rate,
            "fusion_method": self.fusion_method,
            "sequence_length": self.config.sequence_length,
            "enable_window_shift_augmentation": self.config.enable_window_shift_augmentation,
            "gaussian_noise_std": self.gaussian_noise_std,
            "window_shift_window_seconds": self.config.window_shift_window_seconds,
            "window_shift_start_min_seconds": self.config.window_shift_start_min_seconds,
            "window_shift_start_max_seconds": self.config.window_shift_start_max_seconds,
            "window_shift_step_seconds": self.config.window_shift_step_seconds,
            "sampling_rate_hz": self.config.sampling_rate_hz,
            "n_way": self.config.n_way,
            "task_class_ids": list(self.config.task_class_ids),
            "k_shot": self.config.k_shot,
            "q_query": self.config.q_query,
            "task_normalize_mode": self.config.task_normalize_mode,
            "classifier_mode": self.config.classifier_mode,
            "supcon_loss_weight": self.supcon_loss_weight,
            "supcon_temperature": self.supcon_temperature,
            "triplet_loss_weight": self.triplet_loss_weight,
            "triplet_margin": self.triplet_margin,
            "train_batch_size": self.train_batch_size,
            "num_epochs": self.config.num_epochs,
            "tasks_per_epoch": self.config.tasks_per_epoch,
            "val_tasks": self.config.val_tasks,
            "heldout_eval_tasks": self.config.heldout_eval_tasks,
            "k_shot_adaptation_steps": self.config.k_shot_adaptation_steps,
            "train_log_every": self.config.train_log_every,
            "eval_log_every": self.config.eval_log_every,
            "val_batch_size": self.config.val_batch_size,
            "val_every_n_train_steps": self.config.val_every_n_train_steps,
            "summary_every_n_train_steps": self.config.summary_every_n_train_steps,
            "logging_verbosity": self.logging_verbosity,
            "train_prefetch_batches": self.train_prefetch_batches,
            "train_progress_write_every_n_batches": self.config.train_progress_write_every_n_batches,
            "csv_flush_every_events": self.config.csv_flush_every_events,
            "embedding_dim": self.embedding_dim,
            "num_tcn_blocks": self.config.num_tcn_blocks,
            "tcn_dilation_rates": self.config.tcn_dilation_rates,
            "tcn_kernel_size": self.config.tcn_kernel_size,
            "tcn_dropout_rate": self.config.tcn_dropout_rate,
            "tcn_attention_heads": self.config.tcn_attention_heads,
            "tcn_attention_key_dim": self.config.tcn_attention_key_dim,
            "tcn_attention_dropout": self.config.tcn_attention_dropout,
            "tcn_attention_pool_size": self.config.tcn_attention_pool_size,
            "fusion_transformer_heads": self.config.fusion_transformer_heads,
            "fusion_transformer_layers": self.config.fusion_transformer_layers,
            "fusion_transformer_ffn_dim": self.config.fusion_transformer_ffn_dim,
            "fusion_ib_beta": self.config.fusion_ib_beta,
            "clear_session_per_fold": self.config.clear_session_per_fold,
            "single_loso_fold": self.config.single_loso_fold,
            "single_loso_test_subject": self.config.single_loso_test_subject,
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
        self.logger.info(
            f"Logging verbosity={self.logging_verbosity} (0=minimal, 1=standard, 2=detailed)"
        )

    def _augment_training_inputs(self, x: tf.Tensor) -> tf.Tensor:
        """Apply training-only signal augmentation configured for episodic updates."""
        if self.gaussian_noise_std <= 0:
            return x
        noise = tf.random.normal(
            shape=tf.shape(x),
            stddev=tf.cast(self.gaussian_noise_std, x.dtype),
            dtype=x.dtype,
        )
        return x + noise

    def _rebuild_model(self, clear_session: bool = True) -> None:
        """Build a fresh model/optimizer, optionally clearing stale TF graph state."""
        if clear_session:
            tf.keras.backend.clear_session()
            gc.collect()

        # Keep runtime dimensions in sync with dataset-dependent config updates.
        self.sequence_length = int(self.config.sequence_length)
        self.num_sensors = int(self.config.num_sensors)

        self.model = MultimodalPrototypicalNetwork(
            sequence_length=self.config.sequence_length,
            num_sensors=self.num_sensors,
            num_classes=self.config.n_way,
            embedding_dim=self.embedding_dim,
            modality_names=self.config.modality_names,
            fusion_method=self.fusion_method,
            distance_metric=self.distance_metric,
            classifier_mode=self.config.classifier_mode,
            num_tcn_blocks=self.config.num_tcn_blocks,
            tcn_dilation_rates=self.config.tcn_dilation_rates,
            tcn_kernel_size=self.config.tcn_kernel_size,
            filters_list=self.config.filters_list,
            strides=self.config.strides,
            pooling_size=self.config.pooling_size,
            tcn_dropout_rate=self.config.tcn_dropout_rate,
            tcn_attention_heads=self.config.tcn_attention_heads,
            tcn_attention_key_dim=self.config.tcn_attention_key_dim,
            tcn_attention_dropout=self.config.tcn_attention_dropout,
            tcn_attention_pool_size=self.config.tcn_attention_pool_size,
            fusion_transformer_heads=self.config.fusion_transformer_heads,
            fusion_transformer_layers=self.config.fusion_transformer_layers,
            fusion_transformer_ffn_dim=self.config.fusion_transformer_ffn_dim,
            fusion_ib_beta=self.config.fusion_ib_beta,
        )
        self.optimizer = keras.optimizers.AdamW(learning_rate=self.learning_rate)
        self._build_compiled_train_batch_step()

    def _build_compiled_train_batch_step(self) -> None:
        """Build a compiled train-step function bound to current model/optimizer vars."""
        self._compiled_train_batch_step = tf.function(
            self._train_batch_step_compiled_impl,
            reduce_retracing=True,
            input_signature=[
                tf.TensorSpec(
                    shape=(
                        None,
                        self.support_size,
                        self.sequence_length,
                        self.num_sensors,
                    ),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(shape=(None, self.support_size), dtype=tf.int32),
                tf.TensorSpec(
                    shape=(
                        None,
                        self.query_size,
                        self.sequence_length,
                        self.num_sensors,
                    ),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(shape=(None, self.query_size), dtype=tf.int32),
            ],
        )

    def _train_batch_step_compiled_impl(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compiled optimizer update over one batch of episodic tasks."""
        with tf.GradientTape() as tape:

            def _per_task_step(inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]):
                support_x, support_y, query_x, query_y = inputs
                support_x = self._augment_training_inputs(support_x)
                query_x = self._augment_training_inputs(query_x)

                task_outputs = self._forward_task(
                    support_x=support_x,
                    support_y=support_y,
                    query_x=query_x,
                    query_y=query_y,
                    training=True,
                )
                logits = task_outputs["logits"]
                loss = tf.cast(task_outputs["loss"], tf.float32)
                task_loss = tf.cast(task_outputs["task_loss"], tf.float32)
                contrastive_loss = tf.cast(
                    task_outputs["contrastive_loss"],
                    tf.float32,
                )
                predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
                accuracy = tf.reduce_mean(
                    tf.cast(
                        tf.equal(predictions, tf.cast(query_y, tf.int64)),
                        tf.float32,
                    )
                )
                return loss, task_loss, accuracy, contrastive_loss

            losses, task_losses, accuracies, contrastive_losses = tf.map_fn(
                _per_task_step,
                (support_x_batch, support_y_batch, query_x_batch, query_y_batch),
                fn_output_signature=(
                    tf.TensorSpec(shape=(), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.float32),
                ),
                parallel_iterations=16,
            )

            batch_loss = tf.reduce_mean(losses)
            batch_task_loss = tf.reduce_mean(task_losses)
            batch_acc = tf.reduce_mean(accuracies)
            batch_contrastive_loss = tf.reduce_mean(contrastive_losses)

        gradients = tape.gradient(batch_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return batch_loss, batch_task_loss, batch_acc, batch_contrastive_loss

    def _compute_model_aux_loss(self, dtype: tf.dtypes.DType) -> tf.Tensor:
        """Return regularization losses added by submodules, or zero if absent."""
        if not self.model.losses:
            return tf.constant(0.0, dtype=dtype)
        return tf.add_n(self.model.losses)

    def _compute_supervised_contrastive_loss(
        self, embeddings: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        """Compute a label-aware contrastive loss over one episode."""
        if self.supcon_loss_weight <= 0:
            return tf.constant(0.0, dtype=embeddings.dtype)

        normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        logits = tf.matmul(
            normalized_embeddings, normalized_embeddings, transpose_b=True
        )
        logits = logits / tf.cast(self.supcon_temperature, logits.dtype)

        batch_size = tf.shape(labels)[0]
        labels = tf.reshape(labels, (-1, 1))
        positive_mask = tf.cast(tf.equal(labels, tf.transpose(labels)), logits.dtype)
        logits_mask = tf.ones_like(positive_mask) - tf.eye(
            batch_size, dtype=logits.dtype
        )
        positive_mask = positive_mask * logits_mask

        logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)
        exp_logits = tf.exp(logits) * logits_mask
        log_prob = logits - tf.math.log(
            tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-8
        )

        positive_counts = tf.reduce_sum(positive_mask, axis=1)
        mean_log_prob_pos = tf.math.divide_no_nan(
            tf.reduce_sum(positive_mask * log_prob, axis=1),
            positive_counts,
        )
        valid_anchor_mask = tf.cast(positive_counts > 0, logits.dtype)

        return -tf.math.divide_no_nan(
            tf.reduce_sum(mean_log_prob_pos * valid_anchor_mask),
            tf.reduce_sum(valid_anchor_mask),
        )

    def _compute_batch_all_triplet_loss(
        self, embeddings: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        """Compute BatchAllTripletLoss using cosine distance d(a, b)=1-cos(a, b)."""
        if self.triplet_loss_weight <= 0:
            return tf.constant(0.0, dtype=embeddings.dtype)

        normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        pairwise_similarity = tf.matmul(
            normalized_embeddings, normalized_embeddings, transpose_b=True
        )
        pairwise_distance = 1.0 - pairwise_similarity

        labels = tf.reshape(labels, [-1])
        same_label = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        different_label = tf.logical_not(same_label)
        indices_not_equal = tf.logical_not(
            tf.eye(tf.shape(labels)[0], dtype=tf.bool)
        )
        positive_mask = tf.logical_and(same_label, indices_not_equal)

        anchor_positive_dist = tf.expand_dims(pairwise_distance, 2)
        anchor_negative_dist = tf.expand_dims(pairwise_distance, 1)
        triplet_loss = (
            anchor_positive_dist
            - anchor_negative_dist
            + tf.cast(self.triplet_margin, pairwise_distance.dtype)
        )

        mask_ap = tf.expand_dims(positive_mask, 2)
        mask_an = tf.expand_dims(different_label, 1)
        triplet_mask = tf.logical_and(mask_ap, mask_an)
        triplet_loss = tf.where(
            triplet_mask,
            triplet_loss,
            tf.zeros_like(triplet_loss),
        )
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        positive_triplets = tf.cast(triplet_loss > 1e-16, pairwise_distance.dtype)
        return tf.math.divide_no_nan(
            tf.reduce_sum(triplet_loss),
            tf.reduce_sum(positive_triplets),
        )

    def _forward_task(
        self,
        support_x: tf.Tensor,
        support_y: tf.Tensor,
        query_x: tf.Tensor,
        query_y: tf.Tensor,
        training: bool,
        return_similarity_scores: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Run one task and compute classification plus optional embedding losses."""
        episode_outputs = self.model.forward_episode(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            training=training,
        )
        logits = episode_outputs["logits"]
        task_loss = self.loss_fn(query_y, logits)
        model_aux_loss = self._compute_model_aux_loss(dtype=task_loss.dtype)

        embeddings = tf.concat(
            [episode_outputs["support_embeddings"], episode_outputs["query_embeddings"]],
            axis=0,
        )
        labels = tf.concat([support_y, query_y], axis=0)
        total_aux_loss = model_aux_loss
        contrastive_loss = tf.constant(0.0, dtype=task_loss.dtype)
        triplet_loss = tf.constant(0.0, dtype=task_loss.dtype)
        if self.supcon_loss_weight > 0:
            contrastive_loss = (
                tf.cast(self.supcon_loss_weight, task_loss.dtype)
                * self._compute_supervised_contrastive_loss(embeddings, labels)
            )
            total_aux_loss += contrastive_loss
        if self.triplet_loss_weight > 0:
            triplet_loss = (
                tf.cast(self.triplet_loss_weight, task_loss.dtype)
                * self._compute_batch_all_triplet_loss(embeddings, labels)
            )
            total_aux_loss += triplet_loss
        total_loss = task_loss + total_aux_loss

        outputs = {
            "logits": logits,
            "loss": total_loss,
            "task_loss": task_loss,
            "contrastive_loss": contrastive_loss,
            "triplet_loss": triplet_loss,
            "model_aux_loss": model_aux_loss,
            "support_embeddings": episode_outputs["support_embeddings"],
            "query_embeddings": episode_outputs["query_embeddings"],
            "prototypes": episode_outputs["prototypes"],
        }
        if return_similarity_scores:
            outputs["similarity_scores"] = self.model.compute_similarity_scores(
                episode_outputs["query_embeddings"], episode_outputs["prototypes"]
            )
        return outputs

    def train_step(self, support_x, support_y, query_x, query_y):
        """Single training step on one task."""
        with tf.GradientTape() as tape:
            task_outputs = self._forward_task(
                support_x=support_x,
                support_y=support_y,
                query_x=query_x,
                query_y=query_y,
                training=True,
            )
            logits = task_outputs["logits"]
            loss = task_outputs["loss"]

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Compute accuracy
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predictions, tf.cast(query_y, tf.int64)), tf.float32)
        )

        return loss, accuracy

    @staticmethod
    def _stack_task_batch_numpy(
        task_batch: list[dict],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pack a Python task list into dense NumPy arrays once per update."""
        support_x_np = np.stack(
            [task_dict["support_X"] for task_dict in task_batch], axis=0
        )
        support_y_np = np.stack(
            [task_dict["support_y"] for task_dict in task_batch], axis=0
        )
        query_x_np = np.stack([task_dict["query_X"] for task_dict in task_batch], axis=0)
        query_y_np = np.stack([task_dict["query_y"] for task_dict in task_batch], axis=0)
        return support_x_np, support_y_np, query_x_np, query_y_np

    @staticmethod
    def _stack_task_batch(
        task_batch: list[dict],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Pack a Python task list into dense batch tensors once per update."""
        support_x_np, support_y_np, query_x_np, query_y_np = (
            FewShotPainLearner._stack_task_batch_numpy(task_batch)
        )
        return (
            tf.convert_to_tensor(support_x_np, dtype=tf.float32),
            tf.convert_to_tensor(support_y_np, dtype=tf.int32),
            tf.convert_to_tensor(query_x_np, dtype=tf.float32),
            tf.convert_to_tensor(query_y_np, dtype=tf.int32),
        )

    @staticmethod
    def _sample_and_stack_task_batch_numpy(
        sampler,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample one task batch and pack it into dense NumPy arrays."""
        task_batch = [sampler.get_task() for _ in range(max(1, int(batch_size)))]
        return FewShotPainLearner._stack_task_batch_numpy(task_batch)

    def _iter_prefetched_task_batches(
        self,
        sampler,
        tasks_per_epoch: int,
    ):
        """Yield `(batch_size, stacked_numpy_batch)` with async CPU prefetch."""
        batch_sizes = [
            min(self.train_batch_size, tasks_per_epoch - task_start)
            for task_start in range(0, tasks_per_epoch, self.train_batch_size)
        ]
        if not batch_sizes:
            return

        prefetch_batches = max(1, int(self.train_prefetch_batches))
        if prefetch_batches <= 1:
            for batch_size in batch_sizes:
                yield batch_size, self._sample_and_stack_task_batch_numpy(
                    sampler,
                    batch_size,
                )
            return

        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = deque()
            next_batch_idx = 0

            while (
                next_batch_idx < len(batch_sizes)
                and len(pending) < prefetch_batches
            ):
                batch_size = batch_sizes[next_batch_idx]
                pending.append(
                    (
                        batch_size,
                        executor.submit(
                            self._sample_and_stack_task_batch_numpy,
                            sampler,
                            batch_size,
                        ),
                    )
                )
                next_batch_idx += 1

            while pending:
                batch_size, batch_future = pending.popleft()
                batch_arrays = batch_future.result()

                if next_batch_idx < len(batch_sizes):
                    next_size = batch_sizes[next_batch_idx]
                    pending.append(
                        (
                            next_size,
                            executor.submit(
                                self._sample_and_stack_task_batch_numpy,
                                sampler,
                                next_size,
                            ),
                        )
                    )
                    next_batch_idx += 1

                yield batch_size, batch_arrays

    def _train_batch_step_eager_tensors(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Eager fallback for one optimizer update using a batch of tasks."""
        with tf.GradientTape() as tape:
            losses = []
            task_losses = []
            accuracies = []
            contrastive_losses = []
            for support_x, support_y, query_x, query_y in zip(
                tf.unstack(support_x_batch, axis=0),
                tf.unstack(support_y_batch, axis=0),
                tf.unstack(query_x_batch, axis=0),
                tf.unstack(query_y_batch, axis=0),
            ):
                support_x = self._augment_training_inputs(support_x)
                query_x = self._augment_training_inputs(query_x)

                task_outputs = self._forward_task(
                    support_x=support_x,
                    support_y=support_y,
                    query_x=query_x,
                    query_y=query_y,
                    training=True,
                )
                logits = task_outputs["logits"]
                loss = task_outputs["loss"]
                task_loss = task_outputs["task_loss"]
                contrastive_loss = task_outputs["contrastive_loss"]
                predictions = tf.argmax(logits, axis=1)
                accuracy = tf.reduce_mean(
                    tf.cast(
                        tf.equal(predictions, tf.cast(query_y, tf.int64)), tf.float32
                    )
                )
                losses.append(loss)
                task_losses.append(task_loss)
                accuracies.append(accuracy)
                contrastive_losses.append(contrastive_loss)

            batch_loss = tf.reduce_mean(tf.stack(losses))
            batch_task_loss = tf.reduce_mean(tf.stack(task_losses))
            batch_acc = tf.reduce_mean(tf.stack(accuracies))
            batch_contrastive_loss = tf.reduce_mean(tf.stack(contrastive_losses))

        gradients = tape.gradient(batch_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return batch_loss, batch_task_loss, batch_acc, batch_contrastive_loss

    def _train_batch_step_tensors(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Run compiled train step, with eager fallback if compilation fails."""
        if self._compiled_train_batch_step is not None:
            try:
                return self._compiled_train_batch_step(
                    support_x_batch,
                    support_y_batch,
                    query_x_batch,
                    query_y_batch,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.logger.warning(
                    "Compiled train step failed once; falling back to eager for this batch. "
                    f"error={exc!r}"
                )
                self._compiled_train_batch_step = None
        return self._train_batch_step_eager_tensors(
            support_x_batch=support_x_batch,
            support_y_batch=support_y_batch,
            query_x_batch=query_x_batch,
            query_y_batch=query_y_batch,
        )

    @staticmethod
    def _task_batch_has_uniform_shapes(task_batch: list[dict]) -> bool:
        """Return True when support/query tensors share identical shapes across tasks."""
        if not task_batch:
            return False
        keys = ("support_X", "support_y", "query_X", "query_y")
        reference_shapes = {
            key: tuple(np.asarray(task_batch[0][key]).shape) for key in keys
        }
        for task_dict in task_batch[1:]:
            for key in keys:
                if tuple(np.asarray(task_dict[key]).shape) != reference_shapes[key]:
                    return False
        return True

    def train_batch_step(
        self, task_batch: list[dict]
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Single optimizer update using a batch of tasks."""
        (
            support_x_batch,
            support_y_batch,
            query_x_batch,
            query_y_batch,
        ) = self._stack_task_batch(task_batch)
        return self._train_batch_step_tensors(
            support_x_batch=support_x_batch,
            support_y_batch=support_y_batch,
            query_x_batch=query_x_batch,
            query_y_batch=query_y_batch,
        )

    def evaluate_task(self, support_x, support_y, query_x, query_y):
        """Evaluate on one task without updating weights."""
        task_outputs = self._forward_task(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            training=False,
        )
        logits = task_outputs["logits"]
        loss = task_outputs["loss"]

        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predictions, tf.cast(query_y, tf.int64)), tf.float32)
        )

        return loss, accuracy

    def evaluate_batch_step(
        self, task_batch: list[dict]
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Evaluate a batch of tasks without updating weights."""
        batch_loss, metrics = self._evaluate_task_batch_loss_and_metrics(
            task_batch,
        )
        return (
            tf.constant(batch_loss, dtype=tf.float32),
            tf.constant(metrics["accuracy"], dtype=tf.float32),
            tf.constant(metrics["contrastive_loss"], dtype=tf.float32),
        )

    @staticmethod
    def _split_similarity_scores(
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
    def _compute_similarity_metrics(
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

    def _evaluate_task_batch_loss_and_metrics(
        self,
        task_batch: list[dict],
    ) -> tuple[float, dict]:
        """Evaluate a task batch and aggregate classification/similarity metrics."""
        losses = []
        task_losses = []
        contrastive_losses = []
        all_true_tensors = []
        all_pred_tensors = []
        all_intra_class_scores = []
        all_inter_class_scores = []
        class_ids = tf.range(int(self.config.n_way), dtype=tf.int32)[tf.newaxis, :]
        if self._task_batch_has_uniform_shapes(task_batch):
            (
                support_x_batch,
                support_y_batch,
                query_x_batch,
                query_y_batch,
            ) = self._stack_task_batch(task_batch)
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

            task_outputs = self._forward_task(
                support_x,
                support_y,
                query_x,
                query_y,
                training=False,
                return_similarity_scores=True,
            )
            logits = task_outputs["logits"]
            similarity_scores = task_outputs["similarity_scores"]
            loss = task_outputs["loss"]
            task_loss = task_outputs["task_loss"]
            pred = tf.argmax(logits, axis=1, output_type=tf.int32)
            row_indices = tf.range(tf.shape(query_y)[0], dtype=tf.int32)
            intra_class_scores = tf.gather_nd(
                similarity_scores,
                tf.stack([row_indices, query_y], axis=1),
            )
            inter_class_mask = tf.not_equal(class_ids, query_y[:, tf.newaxis])
            inter_class_scores = tf.boolean_mask(similarity_scores, inter_class_mask)

            losses.append(tf.cast(loss, tf.float32))
            task_losses.append(tf.cast(task_loss, tf.float32))
            contrastive_losses.append(tf.cast(task_outputs["contrastive_loss"], tf.float32))
            all_true_tensors.append(query_y)
            all_pred_tensors.append(pred)
            all_intra_class_scores.append(intra_class_scores)
            all_inter_class_scores.append(inter_class_scores)

        y_true = tf.concat(all_true_tensors, axis=0).numpy().astype(np.int32, copy=False)
        y_pred = tf.concat(all_pred_tensors, axis=0).numpy().astype(np.int32, copy=False)
        metrics = self._compute_macro_metrics(y_true, y_pred)
        metrics.update(
            self._compute_similarity_metrics(
                tf.concat(all_intra_class_scores, axis=0).numpy(),
                tf.concat(all_inter_class_scores, axis=0).numpy(),
            )
        )
        metrics["task_loss"] = float(tf.reduce_mean(tf.stack(task_losses)))
        metrics["contrastive_loss"] = float(tf.reduce_mean(tf.stack(contrastive_losses)))
        return float(tf.reduce_mean(tf.stack(losses))), metrics

    def _compute_macro_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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

    def _evaluate_sampler_loss_and_metrics(
        self, sampler, num_tasks: int
    ) -> tuple[float, dict]:
        """Evaluate average loss plus classification/similarity metrics on tasks."""
        return self._evaluate_task_batch_loss_and_metrics(
            [sampler.get_task() for _ in range(num_tasks)],
        )

    def _save_model_architecture(self, sample_task: dict, output_path: str) -> str:
        """Build model and save model architecture summaries to a text file."""
        support_x = tf.constant(sample_task["support_X"], dtype=tf.float32)
        support_y = tf.constant(sample_task["support_y"], dtype=tf.int32)
        query_x = tf.constant(sample_task["query_X"], dtype=tf.float32)
        _ = self.model(support_x, support_y, query_x, training=False)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fp:
            fp.write("=== MultimodalPrototypicalNetwork Summary ===\n")
            full_summary = io.StringIO()
            self.model.summary(print_fn=lambda line: full_summary.write(line + "\n"))
            fp.write(full_summary.getvalue())
            fp.write("\n")

            fp.write("=== Modality Encoder Summaries ===\n")
            for modality_name, encoder in self.model.modality_encoders.items():
                fp.write(f"\n--- Encoder: {modality_name} ---\n")
                encoder_summary = io.StringIO()
                encoder.summary(
                    print_fn=lambda line: encoder_summary.write(line + "\n")
                )
                fp.write(encoder_summary.getvalue())
        print(self.model.summary())
        return output_path

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m {secs}s"
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    def _log_composite_summary(
        self,
        *,
        prefix: str,
        train_metrics: dict,
        val_metrics: dict,
        heldout_metrics: dict,
        elapsed_seconds: float,
    ) -> None:
        composite_accuracy = float(
            np.mean(
                [
                    train_metrics["accuracy"],
                    val_metrics["accuracy"],
                    heldout_metrics["accuracy"],
                ]
            )
        )
        self.logger.info(
            f"{prefix}: "
            f"composite={composite_accuracy:.4f}, "
            f"train_acc={train_metrics['accuracy']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"heldout_acc={heldout_metrics['accuracy']:.4f}, "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"heldout_loss={heldout_metrics['loss']:.4f}, "
            f"elapsed_seconds={elapsed_seconds:.2f}"
        )

    def train(
        self,
        training_progress_output_dir: str = "outputs/training_progress",
        save_model_architecture_first_run: bool = True,
        model_architecture_output_path: str = "outputs/model_architecture/model_summary.txt",
    ):
        """
        Train on all subjects using leave-one-subject-out cross-validation.
        """
        num_epochs = max(1, int(self.config.num_epochs))
        tasks_per_epoch = max(1, int(self.config.tasks_per_epoch))
        val_tasks = max(1, int(self.config.val_tasks))
        heldout_eval_tasks = max(1, int(self.config.heldout_eval_tasks))
        k_shot_adaptation_steps = max(1, int(self.config.k_shot_adaptation_steps))
        train_log_every = max(1, int(self.config.train_log_every))
        eval_log_every = max(1, int(self.config.eval_log_every))
        val_batch_size = max(1, int(self.config.val_batch_size))
        val_every_n_train_steps = max(1, int(self.config.val_every_n_train_steps))

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
            "zero_shot_intra_class_similarities": [],
            "zero_shot_inter_class_similarities": [],
            "k_shot_losses": [],
            "k_shot_accuracies": [],
            "k_shot_precisions": [],
            "k_shot_recalls": [],
            "k_shot_f1s": [],
            "k_shot_intra_class_similarities": [],
            "k_shot_inter_class_similarities": [],
            "training_progress_files": [],
            "model_architecture_file": None,
        }

        if self.config.single_loso_fold:
            if self.config.single_loso_test_subject is not None:
                selected_subject = self.config.single_loso_test_subject
                if selected_subject not in self.cv.subjects:
                    raise ValueError(
                        f"single_loso_test_subject={selected_subject} is not in available subjects."
                    )
                fold_subjects = [selected_subject]
            else:
                fold_subjects = [self.cv.subjects[0]]
            self.logger.info(
                f"single_loso_fold=True: running only one fold with held-out subject={fold_subjects[0]}"
            )
        else:
            fold_subjects = self.cv.subjects

        num_subjects = len(fold_subjects)
        total_train_steps = num_subjects * num_epochs * max(
            1, int(np.ceil(tasks_per_epoch / self.train_batch_size))
        )
        completed_train_steps = 0
        train_start_time = time.perf_counter()
        summary_every_n_train_steps = max(
            1, int(self.config.summary_every_n_train_steps)
        )
        train_progress_write_every_n_batches = max(
            1, int(self.config.train_progress_write_every_n_batches)
        )
        progress = TrainingProgressReporter(
            logger=self.logger,
            train_log_every=train_log_every,
            eval_log_every=eval_log_every,
        )
        csv_writer = TrainingProgressCSVWriter(
            output_dir=training_progress_output_dir,
            flush_every_events=max(1, int(self.config.csv_flush_every_events)),
        )
        architecture_saved = False

        for fold, test_subject in enumerate(fold_subjects):
            fold_start_time = time.perf_counter()
            progress.log_fold_start(
                fold_idx=fold + 1, total_folds=num_subjects, test_subject=test_subject
            )
            progress_file = csv_writer.start_fold(
                fold_idx=fold + 1, test_subject=test_subject
            )

            # Reset model for each fold and free memory from prior graph state.
            self._rebuild_model(
                clear_session=self.config.clear_session_per_fold,
            )

            # Get fold dictionary with samplers
            fold_dict = self.cv.get_fold(test_subject)

            train_sampler = fold_dict["train_sampler"]
            val_sampler = fold_dict["val_sampler"]
            test_sampler = fold_dict["test_sampler"]

            if save_model_architecture_first_run and not architecture_saved:
                sample_task = train_sampler.get_task()
                architecture_path = self._save_model_architecture(
                    sample_task=sample_task,
                    output_path=model_architecture_output_path,
                )
                cv_results["model_architecture_file"] = architecture_path
                architecture_saved = True
                self.logger.info(f"Saved model architecture to {architecture_path}")

            fold_results = {
                "train_losses": [],
                "train_accuracies": [],
                "val_losses": [],
                "val_accuracies": [],
            }
            fold_summary_reference = {
                "train": None,
                "val": None,
                "heldout": None,
            }

            for epoch in range(num_epochs):
                epoch_start_time = time.perf_counter()
                # Training
                epoch_train_losses = []
                epoch_train_accs = []
                epoch_val_losses = []
                epoch_val_accs = []
                processed_tasks = 0
                processed_batches = 0

                for current_batch_size, (
                    support_x_np,
                    support_y_np,
                    query_x_np,
                    query_y_np,
                ) in self._iter_prefetched_task_batches(
                    train_sampler,
                    tasks_per_epoch,
                ):
                    loss, task_loss, acc, contrastive_loss = self._train_batch_step_tensors(
                        support_x_batch=tf.convert_to_tensor(
                            support_x_np,
                            dtype=tf.float32,
                        ),
                        support_y_batch=tf.convert_to_tensor(
                            support_y_np,
                            dtype=tf.int32,
                        ),
                        query_x_batch=tf.convert_to_tensor(
                            query_x_np,
                            dtype=tf.float32,
                        ),
                        query_y_batch=tf.convert_to_tensor(
                            query_y_np,
                            dtype=tf.int32,
                        ),
                    )
                    processed_tasks += current_batch_size
                    processed_batches += 1

                    epoch_train_losses.append(float(loss))
                    epoch_train_accs.append(float(acc))
                    completed_train_steps += 1
                    elapsed = time.perf_counter() - train_start_time
                    avg_step_time = elapsed / max(1, completed_train_steps)
                    remaining_steps = max(0, total_train_steps - completed_train_steps)
                    eta_seconds = remaining_steps * avg_step_time
                    if (
                        processed_batches % train_progress_write_every_n_batches == 0
                        or processed_tasks == tasks_per_epoch
                    ):
                        csv_writer.write_event(
                            fold_idx=fold + 1,
                            test_subject=test_subject,
                            event_type="train_update",
                            epoch=epoch + 1,
                            epoch_total=num_epochs,
                            step=processed_tasks,
                            step_total=tasks_per_epoch,
                            loss=float(loss),
                            task_loss=float(task_loss),
                            contrastive_loss=float(contrastive_loss),
                            accuracy=float(acc),
                        )
                    progress.log_step(
                        stage="Train task",
                        fold_idx=fold + 1,
                        total_folds=num_subjects,
                        epoch_idx=epoch + 1,
                        total_epochs=num_epochs,
                        step_idx=processed_tasks,
                        total_steps=tasks_per_epoch,
                        loss=float(loss),
                        metric_value=float(acc),
                        metric_name="accuracy",
                        extra_metrics={
                            "task_loss": float(task_loss),
                            "contrastive_loss": float(contrastive_loss),
                        },
                        log_every=train_log_every,
                    )
                    if self.logging_verbosity >= 1 and (
                        processed_batches % train_log_every == 0
                        or processed_tasks == tasks_per_epoch
                    ):
                        self.logger.info(
                            f"[Fold {fold + 1}/{num_subjects}] "
                            f"[Epoch {epoch + 1}/{num_epochs}] "
                            f"[Train {processed_tasks}/{tasks_per_epoch} tasks] "
                            f"loss={float(loss):.4f}, task_loss={float(task_loss):.4f}, "
                            f"accuracy={float(acc):.4f}, contrastive_loss={float(contrastive_loss):.4f}, "
                            f"elapsed={self._format_seconds(elapsed)}, "
                            f"eta={self._format_seconds(eta_seconds)}"
                        )

                    if processed_batches % summary_every_n_train_steps == 0:
                        summary_train_loss, summary_train_metrics = (
                            self._evaluate_sampler_loss_and_metrics(
                                train_sampler, num_tasks=val_tasks
                            )
                        )
                        summary_val_loss, summary_val_metrics = (
                            self._evaluate_sampler_loss_and_metrics(
                                val_sampler, num_tasks=val_tasks
                            )
                        )
                        summary_heldout_loss, summary_heldout_metrics = (
                            self._evaluate_sampler_loss_and_metrics(
                                test_sampler, num_tasks=heldout_eval_tasks
                            )
                        )
                        fold_summary_reference["train"] = {
                            "accuracy": summary_train_metrics["accuracy"],
                            "loss": summary_train_loss,
                        }
                        fold_summary_reference["val"] = {
                            "accuracy": summary_val_metrics["accuracy"],
                            "loss": summary_val_loss,
                        }
                        fold_summary_reference["heldout"] = {
                            "accuracy": summary_heldout_metrics["accuracy"],
                            "loss": summary_heldout_loss,
                        }
                        self._log_composite_summary(
                            prefix=(
                                f"[Fold {fold + 1}/{num_subjects}] "
                                f"[Epoch {epoch + 1}/{num_epochs}] "
                                f"[Step {processed_batches}/{max(1, int(np.ceil(tasks_per_epoch / self.train_batch_size)))}] "
                                f"Composite summary"
                            ),
                            train_metrics={
                                "accuracy": summary_train_metrics["accuracy"],
                                "loss": summary_train_loss,
                            },
                            val_metrics={
                                "accuracy": summary_val_metrics["accuracy"],
                                "loss": summary_val_loss,
                            },
                            heldout_metrics={
                                "accuracy": summary_heldout_metrics["accuracy"],
                                "loss": summary_heldout_loss,
                            },
                            elapsed_seconds=time.perf_counter() - train_start_time,
                        )

                    should_run_validation = (
                        (processed_batches % val_every_n_train_steps == 0)
                        or (processed_tasks == tasks_per_epoch)
                    )
                    if not should_run_validation:
                        continue

                    validation_start = time.perf_counter()
                    validation_losses = []
                    validation_task_losses = []
                    validation_accs = []
                    validation_contrastive_losses = []
                    validation_intra_class_similarities = []
                    validation_inter_class_similarities = []
                    for val_task_start in range(0, val_tasks, val_batch_size):
                        current_val_batch_size = min(
                            val_batch_size, val_tasks - val_task_start
                        )
                        val_task_batch = [
                            val_sampler.get_task() for _ in range(current_val_batch_size)
                        ]
                        val_loss, val_metrics = self._evaluate_task_batch_loss_and_metrics(
                            val_task_batch,
                        )
                        validation_losses.append(val_loss)
                        validation_task_losses.append(val_metrics["task_loss"])
                        validation_accs.append(val_metrics["accuracy"])
                        validation_contrastive_losses.append(
                            val_metrics["contrastive_loss"]
                        )
                        validation_intra_class_similarities.append(
                            val_metrics["intra_class_similarity"]
                        )
                        validation_inter_class_similarities.append(
                            val_metrics["inter_class_similarity"]
                        )

                    mean_val_loss = float(np.mean(validation_losses))
                    mean_val_acc = float(np.mean(validation_accs))
                    mean_val_task_loss = float(np.mean(validation_task_losses))
                    mean_val_contrastive_loss = float(
                        np.mean(validation_contrastive_losses)
                    )
                    mean_val_intra_class_similarity = float(
                        np.mean(validation_intra_class_similarities)
                    )
                    mean_val_inter_class_similarity = float(
                        np.mean(validation_inter_class_similarities)
                    )
                    mean_val_similarity_margin = (
                        mean_val_intra_class_similarity
                        - mean_val_inter_class_similarity
                    )
                    epoch_val_losses.append(mean_val_loss)
                    epoch_val_accs.append(mean_val_acc)
                    csv_writer.write_event(
                        fold_idx=fold + 1,
                        test_subject=test_subject,
                        event_type="validation_step",
                        epoch=epoch + 1,
                        epoch_total=num_epochs,
                        step=processed_tasks,
                        step_total=tasks_per_epoch,
                        loss=mean_val_loss,
                        task_loss=mean_val_task_loss,
                        contrastive_loss=mean_val_contrastive_loss,
                        accuracy=mean_val_acc,
                        intra_class_similarity=mean_val_intra_class_similarity,
                        inter_class_similarity=mean_val_inter_class_similarity,
                        similarity_margin=mean_val_similarity_margin,
                    )
                    progress.log_step(
                        stage="Validation",
                        fold_idx=fold + 1,
                        total_folds=num_subjects,
                        step_idx=processed_tasks,
                        total_steps=tasks_per_epoch,
                        loss=mean_val_loss,
                        metric_value=mean_val_acc,
                        metric_name="accuracy",
                        extra_metrics={
                            "task_loss": mean_val_task_loss,
                            "contrastive_loss": mean_val_contrastive_loss,
                            "intra_class_similarity": mean_val_intra_class_similarity,
                            "inter_class_similarity": mean_val_inter_class_similarity,
                            "similarity_margin": mean_val_similarity_margin,
                        },
                        log_every=eval_log_every,
                    )
                    self.logger.info(
                        f"[Fold {fold + 1}/{num_subjects}] "
                        f"[Epoch {epoch + 1}/{num_epochs}] "
                        f"[Validation @ train_batch {processed_batches}] "
                        f"[train_task {processed_tasks}/{tasks_per_epoch}] "
                        f"mean_loss={mean_val_loss:.4f}, "
                        f"mean_task_loss={mean_val_task_loss:.4f}, "
                        f"mean_accuracy={mean_val_acc:.4f}, "
                        f"contrastive_loss={mean_val_contrastive_loss:.4f}, "
                        f"intra_class_similarity={mean_val_intra_class_similarity:.4f}, "
                        f"inter_class_similarity={mean_val_inter_class_similarity:.4f}, "
                        f"similarity_margin={mean_val_similarity_margin:.4f}, "
                        f"validation_elapsed={self._format_seconds(time.perf_counter() - validation_start)}"
                    )

                avg_train_loss = np.mean(epoch_train_losses)
                avg_train_acc = np.mean(epoch_train_accs)
                avg_val_loss = (
                    float(np.mean(epoch_val_losses)) if epoch_val_losses else float("nan")
                )
                avg_val_acc = (
                    float(np.mean(epoch_val_accs)) if epoch_val_accs else float("nan")
                )
                epoch_elapsed = time.perf_counter() - epoch_start_time
                fold_elapsed = time.perf_counter() - fold_start_time
                overall_eta_seconds = (
                    (total_train_steps - completed_train_steps)
                    * (elapsed / max(1, completed_train_steps))
                    if completed_train_steps > 0
                    else float("nan")
                )

                fold_results["train_losses"].append(avg_train_loss)
                fold_results["train_accuracies"].append(avg_train_acc)
                fold_results["val_losses"].append(avg_val_loss)
                fold_results["val_accuracies"].append(avg_val_acc)

                if self.logging_verbosity >= 1 or (epoch + 1) % 2 == 0:
                    self.logger.info(
                        f"[Fold {fold + 1}/{num_subjects}] "
                        f"[Epoch {epoch + 1}/{num_epochs}] "
                        f"train_loss={avg_train_loss:.4f}, train_acc={avg_train_acc:.4f}, "
                        f"val_loss={avg_val_loss:.4f}, val_acc={avg_val_acc:.4f}, "
                        f"epoch_elapsed={self._format_seconds(epoch_elapsed)}, "
                        f"fold_elapsed={self._format_seconds(fold_elapsed)}, "
                        f"overall_eta={self._format_seconds(overall_eta_seconds)}"
                    )

                if self.logging_verbosity >= 1 and fold_summary_reference["train"] is not None:
                    self._log_composite_summary(
                        prefix=(
                            f"[Fold {fold + 1}/{num_subjects}] "
                            f"[Epoch {epoch + 1}/{num_epochs}] "
                            f"Epoch checkpoint"
                        ),
                        train_metrics=fold_summary_reference["train"],
                        val_metrics=fold_summary_reference["val"],
                        heldout_metrics=fold_summary_reference["heldout"],
                        elapsed_seconds=time.perf_counter() - train_start_time,
                    )

            # Zero-shot performance (after training on M-1 subjects).
            zero_shot_loss, zero_shot_metrics = self._evaluate_sampler_loss_and_metrics(
                test_sampler, num_tasks=heldout_eval_tasks
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
                intra_class_similarity=zero_shot_metrics["intra_class_similarity"],
                inter_class_similarity=zero_shot_metrics["inter_class_similarity"],
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
                test_sampler, num_tasks=heldout_eval_tasks
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
                intra_class_similarity=k_shot_metrics["intra_class_similarity"],
                inter_class_similarity=k_shot_metrics["inter_class_similarity"],
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
            cv_results["zero_shot_intra_class_similarities"].append(
                zero_shot_metrics["intra_class_similarity"]
            )
            cv_results["zero_shot_inter_class_similarities"].append(
                zero_shot_metrics["inter_class_similarity"]
            )
            cv_results["k_shot_losses"].append(k_shot_loss)
            cv_results["k_shot_accuracies"].append(k_shot_metrics["accuracy"])
            cv_results["k_shot_precisions"].append(k_shot_metrics["precision"])
            cv_results["k_shot_recalls"].append(k_shot_metrics["recall"])
            cv_results["k_shot_f1s"].append(k_shot_metrics["f1"])
            cv_results["k_shot_intra_class_similarities"].append(
                k_shot_metrics["intra_class_similarity"]
            )
            cv_results["k_shot_inter_class_similarities"].append(
                k_shot_metrics["inter_class_similarity"]
            )
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
        self.logger.info(
            "Average Zero-shot Similarities: "
            f"intra_class={np.mean(cv_results['zero_shot_intra_class_similarities']):.4f}, "
            f"inter_class={np.mean(cv_results['zero_shot_inter_class_similarities']):.4f}"
        )
        self.logger.info(
            "Average K-shot Similarities: "
            f"intra_class={np.mean(cv_results['k_shot_intra_class_similarities']):.4f}, "
            f"inter_class={np.mean(cv_results['k_shot_inter_class_similarities']):.4f}"
        )
        self.logger.info(f"{'=' * 60}\n")

        return cv_results
