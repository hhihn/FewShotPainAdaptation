import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import time
from data_loaders.pain_meta_dataset import PainMetaDataset
from data_loaders.loso_cross_validator import LOSOCrossValidator
from data_loaders.pain_ds_config import PainDatasetConfig
from utils.logger import setup_logger
from utils.reproducibility import set_global_reproducibility
from utils.training_progress import TrainingProgressReporter
from utils.training_progress_csv import TrainingProgressCSVWriter
from learner.episodic_learning_engine import EpisodicLearningEngine
from learner.episode_evaluation_service import EpisodeEvaluationService
from learner.heldout_adaptation_service import HeldoutAdaptationService
from learner.loso_training_runner import LosoTrainingRunner
from learner.model_architecture_writer import ModelArchitectureWriter
from learner.task_batch_pipeline import TaskBatchPipeline
from learner.validation_checkpoint import ValidationCheckpointTracker


class FewShotPainLearner:
    """Meta-learning trainer for personalized pain assessment."""

    def __init__(
        self,
        config: PainDatasetConfig,
        data_dir: str = "./dataset/np-dataset",
        learning_rate: float = 1e-3,
        distance_metric: str = "cosine",
    ):
        """
        Args:
            config: PainDatasetConfig instance
            data_dir: Directory containing numpy files
            learning_rate: Outer loop learning rate
        """
        self.config = config
        self.data_dir = data_dir
        self.learning_rate = learning_rate
        self.distance_metric = distance_metric
        self.seed = int(config.seed)
        self.deterministic_ops = bool(config.deterministic_ops)
        self.embedding_dim = config.embedding_dim
        self.train_batch_size = max(1, int(config.train_batch_size))
        self.embedding_batch_size = max(
            1, int(getattr(config, "embedding_batch_size", 1))
        )
        self.triplet_loss_weight = (
            0.0
            if str(config.attention_mode).strip().lower() == "can"
            else float(config.triplet_loss_weight)
        )
        self.triplet_margin = float(config.triplet_margin)
        self.triplet_mining_strategy = str(config.triplet_mining_strategy)
        self.triplet_center_gradient_clip_norm = float(
            config.triplet_center_gradient_clip_norm
        )
        self.attention_mode = str(config.attention_mode)
        self.can_local_loss_weight = float(config.can_local_loss_weight)
        self.can_global_loss_weight = (
            0.0
            if str(config.attention_mode).strip().lower() == "can"
            else float(config.can_global_loss_weight)
        )
        self.gaussian_noise_std = float(config.gaussian_noise_std)
        self.gradient_clip_norm = getattr(config, "gradient_clip_norm", 1.0)
        if self.gradient_clip_norm is not None:
            self.gradient_clip_norm = float(self.gradient_clip_norm)
        self.augmentation_seed_generator = keras.random.SeedGenerator(
            self.seed + 104729
        )
        self.support_size = int(self.config.n_way * self.config.k_shot)
        self.query_size = int(self.config.n_way * self.config.q_query)
        self.num_sensors = int(len(self.config.sensor_idx))
        self.sequence_length = int(self.config.sequence_length)
        self.train_prefetch_batches = max(
            1, int(getattr(self.config, "train_prefetch_batches", 2))
        )
        self._compiled_train_batch_step = None
        self._compiled_eval_batch_step = None
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

        self.task_pipeline = TaskBatchPipeline(
            train_batch_size=self.train_batch_size,
            embedding_batch_size=self.embedding_batch_size,
            train_prefetch_batches=self.train_prefetch_batches,
        )
        self.engine = EpisodicLearningEngine(self)
        self.evaluator = EpisodeEvaluationService(
            config=self.config,
            engine=self.engine,
            task_pipeline=self.task_pipeline,
        )
        self.adaptation_service = HeldoutAdaptationService(
            engine=self.engine,
            evaluator=self.evaluator,
        )
        self.architecture_writer = ModelArchitectureWriter(
            model_getter=lambda: self.model,
        )
        self.training_runner = LosoTrainingRunner(self)

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
            "lr_schedule": self.config.lr_schedule,
            "lr_decay_alpha": self.config.lr_decay_alpha,
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
            "attention_mode": self.config.attention_mode,
            "can_attention_temperature": self.config.can_attention_temperature,
            "can_meta_hidden_dim": self.config.can_meta_hidden_dim,
            "can_local_loss_weight": self.config.can_local_loss_weight,
            "can_global_loss_weight": self.config.can_global_loss_weight,
            "can_transductive_iterations": self.config.can_transductive_iterations,
            "can_transductive_top_k_per_class": self.config.can_transductive_top_k_per_class,
            "can_transductive_min_confidence": self.config.can_transductive_min_confidence,
            "can_support_mode": self.config.can_support_mode,
            "learned_prototype_slots_per_class": self.config.learned_prototype_slots_per_class,
            "prototype_finetune_epochs": self.config.prototype_finetune_epochs,
            "prototype_finetune_tasks_per_epoch": self.config.prototype_finetune_tasks_per_epoch,
            "prototype_phase2_loss_mode": self.config.prototype_phase2_loss_mode,
            "triplet_loss_weight": self.triplet_loss_weight,
            "triplet_margin": self.triplet_margin,
            "triplet_mining_strategy": self.triplet_mining_strategy,
            "triplet_center_gradient_clip_norm": self.triplet_center_gradient_clip_norm,
            "gradient_clip_norm": self.gradient_clip_norm,
            "train_batch_size": self.train_batch_size,
            "embedding_batch_size": self.embedding_batch_size,
            "num_epochs": self.config.num_epochs,
            "tasks_per_epoch": self.config.tasks_per_epoch,
            "val_tasks": self.config.val_tasks,
            "heldout_eval_tasks": self.config.heldout_eval_tasks,
            "k_shot_adaptation_steps": self.config.k_shot_adaptation_steps,
            "train_log_every": self.config.train_log_every,
            "eval_log_every": self.config.eval_log_every,
            "val_batch_size": self.config.val_batch_size,
            "val_every_n_train_steps": self.config.val_every_n_train_steps,
            "validation_checkpoint_metric": self.config.validation_checkpoint_metric,
            "validation_checkpoint_mode": self.config.validation_checkpoint_mode,
            "logging_verbosity": self.logging_verbosity,
            "train_prefetch_batches": self.train_prefetch_batches,
            "train_progress_write_every_n_batches": self.config.train_progress_write_every_n_batches,
            "csv_flush_every_events": self.config.csv_flush_every_events,
            "embedding_dim": self.embedding_dim,
            "eegnet_temporal_filters": self.config.eegnet_temporal_filters,
            "eegnet_depth_multiplier": self.config.eegnet_depth_multiplier,
            "eegnet_separable_filters": self.config.eegnet_separable_filters,
            "eegnet_temporal_kernel_size": self.config.eegnet_temporal_kernel_size,
            "eegnet_separable_kernel_size": self.config.eegnet_separable_kernel_size,
            "eegnet_pool_size_1": self.config.eegnet_pool_size_1,
            "eegnet_pool_size_2": self.config.eegnet_pool_size_2,
            "eegnet_dropout_rate": self.config.eegnet_dropout_rate,
            "eegnet_l2_weight": self.config.eegnet_l2_weight,
            "clear_session_per_fold": self.config.clear_session_per_fold,
            "single_loso_fold": self.config.single_loso_fold,
            "single_loso_test_subject": self.config.single_loso_test_subject,
            "loso_start_index": self.config.loso_start_index,
            "loso_stop_index": self.config.loso_stop_index,
            "sensor_idx": list(self.config.sensor_idx),
            "modality_names": list(self.config.modality_names),
        }
        if self.attention_mode == "can":
            run_config.update(
                {
                    "embedding_projection_enabled": False,
                    "can_global_loss_weight": 0.0,
                }
            )
            for key in (
                "can_global_loss_weight",
                "embedding_dim",
                "embedding_batch_size",
                "triplet_loss_weight",
                "triplet_margin",
                "triplet_mining_strategy",
                "triplet_center_gradient_clip_norm",
            ):
                run_config.pop(key, None)
        else:
            run_config["embedding_projection_enabled"] = True
        self.logger.info(f"Run config: {json.dumps(run_config, sort_keys=True)}")
        if self.config.clear_session_per_fold:
            self.logger.info(
                "clear_session_per_fold=True is a legacy compatibility setting; "
                "LOSO fold resets restore initial model state without clearing Keras sessions."
            )

        self.logger.info(
            f"Initialized FewShotPainLearner with {len(self.cv.subjects)} subjects"
        )
        num_sensors = len(config.sensor_idx)
        self.logger.info(
            f"Data shape: (sequence_length={config.sequence_length}, num_sensors={num_sensors})"
        )
        self.logger.info(f"Modalities: {config.modality_names}")
        self.logger.info("Encoder: EEGNet-style joint sensor encoder")
        self.logger.info(
            f"Logging verbosity={self.logging_verbosity} (0=minimal, 1=standard, 2=detailed)"
        )

    def _augment_training_inputs(self, x: tf.Tensor) -> tf.Tensor:
        """Apply training-only signal augmentation configured for episodic updates."""
        return self.engine.augment_training_inputs(x)

    def _release_model_resources(self, clear_session: bool = True) -> None:
        """Drop TensorFlow model/optimizer references and optionally clear Keras state."""
        self.engine.release_model_resources(clear_session=clear_session)

    def _rebuild_model(self, clear_session: bool = True) -> None:
        """Build a fresh model/optimizer, optionally clearing stale TF graph state."""
        self.engine.rebuild_model(clear_session=clear_session)

    def _reset_model_state_for_new_fold(self) -> None:
        """Restore initial model/optimizer state while reusing compiled functions."""
        self.engine.reset_model_state_for_new_fold()

    def _build_compiled_train_batch_step(self) -> None:
        """Build a compiled train-step function bound to current model/optimizer vars."""
        self.engine.build_compiled_train_batch_step()

    def _build_compiled_eval_batch_step(self) -> None:
        """Build a compiled evaluation function for batches of episodic tasks."""
        self.engine.build_compiled_eval_batch_step()

    def _get_loso_fold_subjects(self) -> list[int]:
        """Return held-out subjects selected by single-fold and LOSO index config."""
        subjects = [int(subject) for subject in self.cv.subjects]
        if not subjects:
            raise ValueError("No LOSO subjects are available.")

        start_index = self.config.loso_start_index
        stop_index = self.config.loso_stop_index
        range_configured = start_index is not None or stop_index is not None

        if self.config.single_loso_fold and not range_configured:
            if self.config.single_loso_test_subject is not None:
                selected_subject = int(self.config.single_loso_test_subject)
                if selected_subject not in subjects:
                    raise ValueError(
                        f"single_loso_test_subject={selected_subject} is not in available subjects."
                    )
                fold_subjects = [selected_subject]
            else:
                fold_subjects = [subjects[0]]
            self.logger.info(
                f"single_loso_fold=True: running only one fold with held-out subject={fold_subjects[0]}"
            )
            return fold_subjects

        if start_index is None and stop_index is None:
            return subjects

        start_offset = 0 if start_index is None else int(start_index) - 1
        stop_offset = len(subjects) if stop_index is None else int(stop_index)
        fold_subjects = subjects[start_offset:stop_offset]
        if not fold_subjects:
            raise ValueError(
                "Configured LOSO index range selected no subjects "
                f"(loso_start_index={start_index}, loso_stop_index={stop_index}, "
                f"available_folds={len(subjects)})."
            )

        self.logger.info(
            "Running LOSO index range "
            f"{start_index or 1}..{stop_index or len(subjects)} "
            f"(1-based inclusive), selected {len(fold_subjects)} of {len(subjects)} folds."
        )
        return fold_subjects

    def _eval_task_batch_step_compiled_impl(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        """Compiled evaluation over a batch of tasks without optimizer updates."""
        return self.engine._eval_task_batch_step_compiled_impl(
            support_x_batch,
            support_y_batch,
            query_x_batch,
            query_y_batch,
        )

    def _train_batch_step_compiled_impl(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        """Compiled optimizer update over one batch of episodic tasks."""
        return self.engine._train_batch_step_compiled_impl(
            support_x_batch,
            support_y_batch,
            query_x_batch,
            query_y_batch,
        )

    def _train_prototype_memory_batch_step_tensors(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        """Run one phase-2 learned-prototype-memory optimizer update."""
        return self.engine.train_prototype_memory_batch_step_tensors(
            support_x_batch=support_x_batch,
            support_y_batch=support_y_batch,
            query_x_batch=query_x_batch,
            query_y_batch=query_y_batch,
        )

    def _resolve_prototype_finetune_tasks_per_epoch(self, train_sampler) -> int:
        """Return the phase-2 update budget for full-subject prototype tasks."""
        configured_tasks = self.config.prototype_finetune_tasks_per_epoch
        if configured_tasks is not None:
            return max(1, int(configured_tasks))
        active_subject_count = int(len(train_sampler.active_subjects_array))
        return max(1, active_subject_count)

    def _compute_model_aux_loss(self, dtype: tf.dtypes.DType) -> tf.Tensor:
        """Return regularization losses added by submodules, or zero if absent."""
        return self.engine.compute_model_aux_loss(dtype)

    def _apply_gradients(self, loss: tf.Tensor, tape: tf.GradientTape) -> tf.Tensor:
        """Apply gradients for the current model update."""
        return self.engine.apply_gradients(loss, tape)

    def _compute_batch_all_triplet_loss(
        self, embeddings: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        """Compute BatchAllTripletLoss using cosine distance d(a, b)=1-cos(a, b)."""
        return self.engine.compute_batch_all_triplet_loss(embeddings, labels)

    def _compute_batch_hard_triplet_loss(
        self, embeddings: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        """Compute BatchHardTripletLoss using cosine distance d(a, b)=1-cos(a, b)."""
        return self.engine.compute_batch_hard_triplet_loss(embeddings, labels)

    def _compute_triplet_center_loss(
        self, embeddings: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        """Compute Triplet-Center Loss using trainable class centers."""
        return self.engine.compute_triplet_center_loss(embeddings, labels)

    def _compute_triplet_loss(
        self, embeddings: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        """Dispatch configured triplet mining strategy over one episode."""
        return self.engine.compute_triplet_loss(embeddings, labels)

    def _compute_task_batch_objective(
        self,
        episode_outputs: dict[str, tf.Tensor],
        support_y_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        """Compute per-task objective tensors for normalized batched episode outputs."""
        return self.engine.compute_task_batch_objective(
            episode_outputs,
            support_y_batch,
            query_y_batch,
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
        return self.engine.forward_task(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            training=training,
            return_similarity_scores=return_similarity_scores,
        )

    def _forward_task_batch(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
        training: bool,
        return_similarity_scores: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Run multiple tasks with batched embedding and per-task losses."""
        return self.engine.forward_task_batch(
            support_x_batch=support_x_batch,
            support_y_batch=support_y_batch,
            query_x_batch=query_x_batch,
            query_y_batch=query_y_batch,
            training=training,
            return_similarity_scores=return_similarity_scores,
        )

    def train_step(self, support_x, support_y, query_x, query_y):
        """Single training step on one task."""
        return self.engine.train_step(support_x, support_y, query_x, query_y)

    @staticmethod
    def _stack_task_batch_numpy(
        task_batch: list[dict],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pack a Python task list into dense NumPy arrays once per update."""
        return TaskBatchPipeline.stack_task_batch_numpy(task_batch)

    @staticmethod
    def _stack_task_batch(
        task_batch: list[dict],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Pack a Python task list into dense batch tensors once per update."""
        return TaskBatchPipeline.stack_task_batch(task_batch)

    @staticmethod
    def _sample_and_stack_task_batch_numpy(
        sampler,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample one task batch and pack it into dense NumPy arrays."""
        return TaskBatchPipeline.sample_and_stack_task_batch_numpy(
            sampler,
            batch_size,
        )

    def _iter_prefetched_task_batches(
        self,
        sampler,
        tasks_per_epoch: int,
    ):
        """Yield `(batch_size, stacked_numpy_batch)` with async CPU prefetch."""
        yield from self.task_pipeline.iter_prefetched_task_batches(
            sampler,
            tasks_per_epoch,
        )

    def _iter_task_tensor_chunks(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ):
        """Yield task tensor chunks sized by embedding_batch_size in eager mode."""
        yield from self.task_pipeline.iter_task_tensor_chunks(
            support_x_batch,
            support_y_batch,
            query_x_batch,
            query_y_batch,
        )

    def _augment_training_task_chunk(
        self,
        support_x_chunk: tf.Tensor,
        query_x_chunk: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply train augmentation while preserving legacy single-task shapes."""
        task_count = int(tf.shape(support_x_chunk)[0].numpy())
        if task_count == 1:
            return (
                self._augment_training_inputs(support_x_chunk[0])[tf.newaxis, ...],
                self._augment_training_inputs(query_x_chunk[0])[tf.newaxis, ...],
            )
        return (
            self._augment_training_inputs(support_x_chunk),
            self._augment_training_inputs(query_x_chunk),
        )

    def _forward_task_chunk(
        self,
        support_x_chunk: tf.Tensor,
        support_y_chunk: tf.Tensor,
        query_x_chunk: tf.Tensor,
        query_y_chunk: tf.Tensor,
        *,
        training: bool,
        return_similarity_scores: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Forward one eager task chunk and normalize outputs to task-major tensors."""
        return self.engine.forward_task_chunk(
            support_x_chunk=support_x_chunk,
            support_y_chunk=support_y_chunk,
            query_x_chunk=query_x_chunk,
            query_y_chunk=query_y_chunk,
            training=training,
            return_similarity_scores=return_similarity_scores,
        )

    @staticmethod
    def _mean_concat(tensor_parts: list[tf.Tensor]) -> tf.Tensor:
        """Mean over rank-1 tensors collected from task chunks."""
        return tf.reduce_mean(tf.concat(tensor_parts, axis=0))

    @staticmethod
    def _train_metric_tensors_from_chunk_outputs(
        chunk_outputs: dict[str, tf.Tensor],
        query_y_chunk: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        """Return per-task train losses and accuracies for one normalized chunk."""
        return EpisodicLearningEngine.train_metric_tensors_from_chunk_outputs(
            chunk_outputs,
            query_y_chunk,
        )

    def _split_batched_similarity_scores(
        self,
        similarity_scores: tf.Tensor,
        query_y_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Split batched query-to-prototype scores into intra/inter-class groups."""
        query_y_batch = tf.cast(query_y_batch, tf.int32)
        task_count = tf.shape(query_y_batch)[0]
        query_size = tf.shape(query_y_batch)[1]
        row_indices = tf.tile(
            tf.range(query_size, dtype=tf.int32)[tf.newaxis, :],
            [task_count, 1],
        )
        task_indices = tf.tile(
            tf.range(task_count, dtype=tf.int32)[:, tf.newaxis],
            [1, query_size],
        )
        intra_class_scores = tf.gather_nd(
            similarity_scores,
            tf.stack([task_indices, row_indices, query_y_batch], axis=2),
        )

        class_ids = tf.range(int(self.config.n_way), dtype=tf.int32)
        inter_class_mask = tf.not_equal(
            class_ids[tf.newaxis, tf.newaxis, :],
            query_y_batch[:, :, tf.newaxis],
        )
        return intra_class_scores, tf.boolean_mask(similarity_scores, inter_class_mask)

    def _eval_metric_tensors_from_chunk_outputs(
        self,
        chunk_outputs: dict[str, tf.Tensor],
        query_y_chunk: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        """Return flattened eval losses, labels, predictions, and similarity groups."""
        return self.engine.eval_metric_tensors_from_chunk_outputs(
            chunk_outputs,
            query_y_chunk,
        )

    def _train_batch_step_eager_tensors(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        """Eager fallback for one optimizer update using a batch of tasks."""
        return self.engine.train_batch_step_eager_tensors(
            support_x_batch=support_x_batch,
            support_y_batch=support_y_batch,
            query_x_batch=query_x_batch,
            query_y_batch=query_y_batch,
        )

    def _train_batch_step_tensors(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        """Run compiled train step, with eager fallback if compilation fails."""
        return self.engine.train_batch_step_tensors(
            support_x_batch=support_x_batch,
            support_y_batch=support_y_batch,
            query_x_batch=query_x_batch,
            query_y_batch=query_y_batch,
        )

    def _eval_task_batch_step_eager_tensors(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        """Eager fallback for task-batch evaluation without optimizer updates."""
        return self.engine.eval_task_batch_step_eager_tensors(
            support_x_batch=support_x_batch,
            support_y_batch=support_y_batch,
            query_x_batch=query_x_batch,
            query_y_batch=query_y_batch,
        )

    def _eval_task_batch_step_tensors(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        """Run compiled task-batch eval, with eager fallback if compilation fails."""
        return self.engine.eval_task_batch_step_tensors(
            support_x_batch=support_x_batch,
            support_y_batch=support_y_batch,
            query_x_batch=query_x_batch,
            query_y_batch=query_y_batch,
        )

    @staticmethod
    def _task_batch_has_uniform_shapes(task_batch: list[dict]) -> bool:
        """Return True when support/query tensors share identical shapes across tasks."""
        return TaskBatchPipeline.task_batch_has_uniform_shapes(task_batch)

    def train_batch_step(
        self, task_batch: list[dict]
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Single optimizer update using a batch of tasks."""
        (
            support_x_batch,
            support_y_batch,
            query_x_batch,
            query_y_batch,
        ) = self._stack_task_batch(task_batch)
        return self.engine.train_batch_step_tensors(
            support_x_batch=support_x_batch,
            support_y_batch=support_y_batch,
            query_x_batch=query_x_batch,
            query_y_batch=query_y_batch,
        )

    def evaluate_task(self, support_x, support_y, query_x, query_y):
        """Evaluate on one task without updating weights."""
        return self.engine.evaluate_task(support_x, support_y, query_x, query_y)

    def evaluate_batch_step(
        self, task_batch: list[dict]
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Evaluate a batch of tasks without updating weights."""
        batch_loss, metrics = self._evaluate_task_batch_loss_and_metrics(
            task_batch,
        )
        return (
            tf.constant(batch_loss, dtype=tf.float32),
            tf.constant(metrics["accuracy"], dtype=tf.float32),
            tf.constant(metrics["contrastive_loss"], dtype=tf.float32),
            tf.constant(metrics["triplet_loss"], dtype=tf.float32),
        )

    @staticmethod
    def _split_similarity_scores(
        similarity_scores: np.ndarray, y_true: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split query-to-prototype similarities into intra/inter-class groups."""
        return EpisodeEvaluationService.split_similarity_scores(
            similarity_scores,
            y_true,
        )

    @staticmethod
    def _compute_similarity_metrics(
        intra_class_scores: np.ndarray, inter_class_scores: np.ndarray
    ) -> dict:
        """Aggregate similarity statistics using the existing metric dict shape."""
        return EpisodeEvaluationService.compute_similarity_metrics(
            intra_class_scores,
            inter_class_scores,
        )

    def _evaluate_task_batch_loss_and_metrics(
        self,
        task_batch: list[dict],
        *,
        forward_batch_size: int | None = None,
    ) -> tuple[float, dict]:
        """Evaluate a task batch and aggregate classification/similarity metrics."""
        return self.evaluator.evaluate_task_batch_loss_and_metrics(
            task_batch,
            forward_batch_size=forward_batch_size,
        )

    def _compute_macro_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute accuracy, macro precision, macro recall, and macro F1."""
        return self.evaluator.compute_macro_metrics(y_true, y_pred)

    def _evaluate_sampler_loss_and_metrics(
        self,
        sampler,
        num_tasks: int,
        *,
        forward_batch_size: int | None = None,
    ) -> tuple[float, dict]:
        """Evaluate average loss plus classification/similarity metrics on tasks."""
        return self.evaluator.evaluate_sampler_loss_and_metrics(
            sampler,
            num_tasks,
            forward_batch_size=forward_batch_size,
        )

    @staticmethod
    def _set_sampler_task_size(sampler, k_shot: int, q_query: int) -> None:
        """Update sampler task size in-place for temporary held-out evaluation sweeps."""
        EpisodeEvaluationService.set_sampler_task_size(
            sampler,
            k_shot=k_shot,
            q_query=q_query,
        )

    def _evaluate_sampler_loss_and_metrics_at_task_size(
        self,
        sampler,
        num_tasks: int,
        *,
        k_shot: int,
        q_query: int,
        forward_batch_size: int | None = None,
    ) -> tuple[float, dict]:
        """Evaluate sampler metrics with a temporary k-shot/q-query override."""
        return self.evaluator.evaluate_sampler_loss_and_metrics_at_task_size(
            sampler,
            num_tasks=num_tasks,
            k_shot=k_shot,
            q_query=q_query,
            forward_batch_size=forward_batch_size,
        )

    def _adapt_on_sampler_at_task_size(
        self,
        sampler,
        *,
        adaptation_steps: int,
        k_shot: int,
        q_query: int,
    ) -> list[float]:
        """Run adaptation on tasks drawn with temporary k-shot/q-query override."""
        return self.adaptation_service.adapt_on_sampler_at_task_size(
            sampler,
            adaptation_steps=adaptation_steps,
            k_shot=k_shot,
            q_query=q_query,
        )

    def _save_model_architecture(self, sample_task: dict, output_path: str) -> str:
        """Build model and save model architecture summaries to a text file."""
        return self.architecture_writer.save_model_architecture(
            sample_task,
            output_path,
        )

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

    def _log_cross_validation_aggregate(
        self,
        cv_results: dict,
        *,
        title: str,
    ) -> None:
        """Log aggregate cross-validation metrics in the same format as final summary."""
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(title)
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
        if cv_results.get("zero_shot_transductive_accuracies"):
            self.logger.info(
                "Average Zero-shot Transductive Accuracy: "
                f"{np.mean(cv_results['zero_shot_transductive_accuracies']):.4f} "
                f"(±{np.std(cv_results['zero_shot_transductive_accuracies']):.4f})"
            )
        if cv_results.get("k_shot_transductive_accuracies"):
            self.logger.info(
                "Average K-shot Transductive Accuracy: "
                f"{np.mean(cv_results['k_shot_transductive_accuracies']):.4f} "
                f"(±{np.std(cv_results['k_shot_transductive_accuracies']):.4f})"
            )
        if cv_results.get("zero_shot_can_score_margins"):
            self.logger.info(
                "Average Zero-shot CAN Scores: "
                f"true_class={np.mean(cv_results['zero_shot_can_true_class_scores']):.4f}, "
                f"best_other={np.mean(cv_results['zero_shot_can_best_other_scores']):.4f}, "
                f"margin={np.mean(cv_results['zero_shot_can_score_margins']):.4f}"
            )
            self.logger.info(
                "Average K-shot CAN Scores: "
                f"true_class={np.mean(cv_results['k_shot_can_true_class_scores']):.4f}, "
                f"best_other={np.mean(cv_results['k_shot_can_best_other_scores']):.4f}, "
                f"margin={np.mean(cv_results['k_shot_can_score_margins']):.4f}"
            )
        elif cv_results.get("zero_shot_intra_class_similarities"):
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

    @staticmethod
    def _append_evaluation_diagnostics(
        bucket: dict,
        prefix: str,
        metrics: dict,
    ) -> None:
        if "can_score_margin" in metrics:
            bucket[f"{prefix}_can_true_class_scores"].append(
                metrics["can_true_class_score"]
            )
            bucket[f"{prefix}_can_best_other_scores"].append(
                metrics["can_best_other_score"]
            )
            bucket[f"{prefix}_can_score_margins"].append(metrics["can_score_margin"])
        else:
            bucket[f"{prefix}_intra_class_similarities"].append(
                metrics["intra_class_similarity"]
            )
            bucket[f"{prefix}_inter_class_similarities"].append(
                metrics["inter_class_similarity"]
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
        return self.training_runner.train(
            training_progress_output_dir=training_progress_output_dir,
            save_model_architecture_first_run=save_model_architecture_first_run,
            model_architecture_output_path=model_architecture_output_path,
        )

    def _run_loso_training_workflow(
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
        k_shot_adaptation_steps = max(0, int(self.config.k_shot_adaptation_steps))
        train_log_every = max(1, int(self.config.train_log_every))
        eval_log_every = max(1, int(self.config.eval_log_every))
        val_batch_size = max(1, int(self.config.val_batch_size))
        val_every_n_train_steps = max(1, int(self.config.val_every_n_train_steps))
        configured_eval_pair = (int(self.config.k_shot), int(self.config.q_query))
        heldout_eval_pairs = [configured_eval_pair, (1, 1), (5, 5), (10, 10)]
        dedup_pairs: list[tuple[int, int]] = []
        for eval_pair in heldout_eval_pairs:
            if eval_pair not in dedup_pairs:
                dedup_pairs.append(eval_pair)
        heldout_eval_pairs = dedup_pairs

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
            "zero_shot_can_true_class_scores": [],
            "zero_shot_can_best_other_scores": [],
            "zero_shot_can_score_margins": [],
            "zero_shot_transductive_losses": [],
            "zero_shot_transductive_accuracies": [],
            "zero_shot_transductive_precisions": [],
            "zero_shot_transductive_recalls": [],
            "zero_shot_transductive_f1s": [],
            "k_shot_losses": [],
            "k_shot_accuracies": [],
            "k_shot_precisions": [],
            "k_shot_recalls": [],
            "k_shot_f1s": [],
            "k_shot_intra_class_similarities": [],
            "k_shot_inter_class_similarities": [],
            "k_shot_can_true_class_scores": [],
            "k_shot_can_best_other_scores": [],
            "k_shot_can_score_margins": [],
            "k_shot_transductive_losses": [],
            "k_shot_transductive_accuracies": [],
            "k_shot_transductive_precisions": [],
            "k_shot_transductive_recalls": [],
            "k_shot_transductive_f1s": [],
            "heldout_eval_task_sizes": [
                {"k_shot": int(k_shot), "q_query": int(q_query)}
                for k_shot, q_query in heldout_eval_pairs
            ],
            "heldout_eval_by_task_size": {
                f"k{k_shot}_q{q_query}": {
                    "k_shot": int(k_shot),
                    "q_query": int(q_query),
                    "zero_shot_losses": [],
                    "zero_shot_accuracies": [],
                    "zero_shot_precisions": [],
                    "zero_shot_recalls": [],
                    "zero_shot_f1s": [],
                    "zero_shot_intra_class_similarities": [],
                    "zero_shot_inter_class_similarities": [],
                    "zero_shot_can_true_class_scores": [],
                    "zero_shot_can_best_other_scores": [],
                    "zero_shot_can_score_margins": [],
                    "zero_shot_transductive_losses": [],
                    "zero_shot_transductive_accuracies": [],
                    "zero_shot_transductive_precisions": [],
                    "zero_shot_transductive_recalls": [],
                    "zero_shot_transductive_f1s": [],
                    "k_shot_losses": [],
                    "k_shot_accuracies": [],
                    "k_shot_precisions": [],
                    "k_shot_recalls": [],
                    "k_shot_f1s": [],
                    "k_shot_intra_class_similarities": [],
                    "k_shot_inter_class_similarities": [],
                    "k_shot_can_true_class_scores": [],
                    "k_shot_can_best_other_scores": [],
                    "k_shot_can_score_margins": [],
                    "k_shot_transductive_losses": [],
                    "k_shot_transductive_accuracies": [],
                    "k_shot_transductive_precisions": [],
                    "k_shot_transductive_recalls": [],
                    "k_shot_transductive_f1s": [],
                }
                for k_shot, q_query in heldout_eval_pairs
            },
            "training_progress_files": [],
            "model_architecture_file": None,
            "validation_checkpoint_metric": self.config.validation_checkpoint_metric,
            "validation_checkpoint_mode": self.config.validation_checkpoint_mode,
            "validation_checkpoint_values": [],
            "validation_checkpoint_epochs": [],
            "validation_checkpoint_steps": [],
            "validation_checkpoint_metrics": [],
        }

        fold_subjects = self._get_loso_fold_subjects()
        num_subjects = len(fold_subjects)
        fold_checkpoint_interval = max(1, int(np.ceil(num_subjects / 10)))
        total_train_steps = (
            num_subjects
            * num_epochs
            * max(1, int(np.ceil(tasks_per_epoch / self.train_batch_size)))
        )
        completed_train_steps = 0
        train_start_time = time.perf_counter()
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

            # Reset fold state without clearing Keras or recreating tf.functions.
            self._reset_model_state_for_new_fold()
            validation_checkpoint = ValidationCheckpointTracker(
                metric=self.config.validation_checkpoint_metric,
                mode=self.config.validation_checkpoint_mode,
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
                    (
                        loss,
                        task_loss,
                        acc,
                        contrastive_loss,
                        triplet_loss,
                        can_local_loss,
                        can_global_loss,
                    ) = self._train_batch_step_tensors(
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
                        can_mode = self.attention_mode == "can"
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
                            contrastive_loss=None if can_mode else float(contrastive_loss),
                            triplet_loss=None if can_mode else float(triplet_loss),
                            can_local_loss=float(can_local_loss),
                            can_global_loss=None if can_mode else float(can_global_loss),
                            accuracy=float(acc),
                        )
                    train_extra_metrics = {
                        "task_loss": float(task_loss),
                        "can_local_loss": float(can_local_loss),
                    }
                    if self.attention_mode != "can":
                        train_extra_metrics.update(
                            {
                                "contrastive_loss": float(contrastive_loss),
                                "triplet_loss": float(triplet_loss),
                                "can_global_loss": float(can_global_loss),
                            }
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
                        extra_metrics=train_extra_metrics,
                        log_every=train_log_every,
                    )
                    if self.logging_verbosity >= 1 and (
                        processed_batches % train_log_every == 0
                        or processed_tasks == tasks_per_epoch
                    ):
                        if self.attention_mode == "can":
                            self.logger.info(
                                f"[Fold {fold + 1}/{num_subjects}] "
                                f"[Epoch {epoch + 1}/{num_epochs}] "
                                f"[Train {processed_tasks}/{tasks_per_epoch} tasks] "
                                f"loss={float(loss):.4f}, task_loss={float(task_loss):.4f}, "
                                f"accuracy={float(acc):.4f}, "
                                f"can_local_loss={float(can_local_loss):.4f}, "
                                f"elapsed={self._format_seconds(elapsed)}, "
                                f"eta={self._format_seconds(eta_seconds)}"
                            )
                        else:
                            self.logger.info(
                                f"[Fold {fold + 1}/{num_subjects}] "
                                f"[Epoch {epoch + 1}/{num_epochs}] "
                                f"[Train {processed_tasks}/{tasks_per_epoch} tasks] "
                                f"loss={float(loss):.4f}, task_loss={float(task_loss):.4f}, "
                                f"accuracy={float(acc):.4f}, contrastive_loss={float(contrastive_loss):.4f}, "
                                f"triplet_loss={float(triplet_loss):.4f}, "
                                f"can_local_loss={float(can_local_loss):.4f}, "
                                f"can_global_loss={float(can_global_loss):.4f}, "
                                f"elapsed={self._format_seconds(elapsed)}, "
                                f"eta={self._format_seconds(eta_seconds)}"
                            )

                    should_run_validation = (
                        processed_batches % val_every_n_train_steps == 0
                    ) or (processed_tasks == tasks_per_epoch)
                    if not should_run_validation:
                        continue

                    validation_start = time.perf_counter()
                    validation_losses = []
                    validation_task_losses = []
                    validation_accs = []
                    validation_precisions = []
                    validation_recalls = []
                    validation_f1s = []
                    validation_contrastive_losses = []
                    validation_triplet_losses = []
                    validation_can_local_losses = []
                    validation_can_global_losses = []
                    validation_intra_class_similarities = []
                    validation_inter_class_similarities = []
                    validation_can_true_class_scores = []
                    validation_can_best_other_scores = []
                    validation_can_score_margins = []
                    for val_task_start in range(0, val_tasks, val_batch_size):
                        current_val_batch_size = min(
                            val_batch_size, val_tasks - val_task_start
                        )
                        val_task_batch = [
                            val_sampler.get_task()
                            for _ in range(current_val_batch_size)
                        ]
                        val_loss, val_metrics = (
                            self._evaluate_task_batch_loss_and_metrics(
                                val_task_batch,
                            )
                        )
                        validation_losses.append(val_loss)
                        validation_task_losses.append(val_metrics["task_loss"])
                        validation_accs.append(val_metrics["accuracy"])
                        validation_precisions.append(val_metrics["precision"])
                        validation_recalls.append(val_metrics["recall"])
                        validation_f1s.append(val_metrics["f1"])
                        validation_contrastive_losses.append(
                            val_metrics["contrastive_loss"]
                        )
                        validation_triplet_losses.append(val_metrics["triplet_loss"])
                        validation_can_local_losses.append(
                            val_metrics["can_local_loss"]
                        )
                        validation_can_global_losses.append(
                            val_metrics["can_global_loss"]
                        )
                        validation_intra_class_similarities.append(
                            val_metrics["intra_class_similarity"]
                        )
                        validation_inter_class_similarities.append(
                            val_metrics["inter_class_similarity"]
                        )
                        if "can_score_margin" in val_metrics:
                            validation_can_true_class_scores.append(
                                val_metrics["can_true_class_score"]
                            )
                            validation_can_best_other_scores.append(
                                val_metrics["can_best_other_score"]
                            )
                            validation_can_score_margins.append(
                                val_metrics["can_score_margin"]
                            )

                    mean_val_loss = float(np.mean(validation_losses))
                    mean_val_acc = float(np.mean(validation_accs))
                    mean_val_precision = float(np.mean(validation_precisions))
                    mean_val_recall = float(np.mean(validation_recalls))
                    mean_val_f1 = float(np.mean(validation_f1s))
                    mean_val_task_loss = float(np.mean(validation_task_losses))
                    mean_val_contrastive_loss = float(
                        np.mean(validation_contrastive_losses)
                    )
                    mean_val_triplet_loss = float(np.mean(validation_triplet_losses))
                    mean_val_can_local_loss = float(
                        np.mean(validation_can_local_losses)
                    )
                    mean_val_can_global_loss = float(
                        np.mean(validation_can_global_losses)
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
                    mean_val_can_true_class_score = (
                        float(np.mean(validation_can_true_class_scores))
                        if validation_can_true_class_scores
                        else None
                    )
                    mean_val_can_best_other_score = (
                        float(np.mean(validation_can_best_other_scores))
                        if validation_can_best_other_scores
                        else None
                    )
                    mean_val_can_score_margin = (
                        float(np.mean(validation_can_score_margins))
                        if validation_can_score_margins
                        else None
                    )
                    epoch_val_losses.append(mean_val_loss)
                    epoch_val_accs.append(mean_val_acc)
                    validation_checkpoint_metrics = {
                        "loss": mean_val_loss,
                        "task_loss": mean_val_task_loss,
                        "contrastive_loss": mean_val_contrastive_loss,
                        "triplet_loss": mean_val_triplet_loss,
                        "can_local_loss": mean_val_can_local_loss,
                        "can_global_loss": mean_val_can_global_loss,
                        "accuracy": mean_val_acc,
                        "precision": mean_val_precision,
                        "recall": mean_val_recall,
                        "f1": mean_val_f1,
                        "intra_class_similarity": mean_val_intra_class_similarity,
                        "inter_class_similarity": mean_val_inter_class_similarity,
                        "similarity_margin": mean_val_similarity_margin,
                    }
                    if mean_val_can_score_margin is not None:
                        validation_checkpoint_metrics.update(
                            {
                                "can_true_class_score": mean_val_can_true_class_score,
                                "can_best_other_score": mean_val_can_best_other_score,
                                "can_score_margin": mean_val_can_score_margin,
                            }
                        )
                    checkpoint_value = validation_checkpoint_metrics[
                        validation_checkpoint.metric
                    ]
                    checkpoint_improved = validation_checkpoint.maybe_update(
                        value=checkpoint_value,
                        epoch=epoch + 1,
                        step=processed_tasks,
                        metrics=validation_checkpoint_metrics,
                        model_variables=self.model.weights,
                        optimizer_variables=self.optimizer.variables,
                    )
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
                        contrastive_loss=None
                        if mean_val_can_score_margin is not None
                        else mean_val_contrastive_loss,
                        triplet_loss=None
                        if mean_val_can_score_margin is not None
                        else mean_val_triplet_loss,
                        can_local_loss=mean_val_can_local_loss,
                        can_global_loss=None
                        if mean_val_can_score_margin is not None
                        else mean_val_can_global_loss,
                        accuracy=mean_val_acc,
                        precision=mean_val_precision,
                        recall=mean_val_recall,
                        f1=mean_val_f1,
                        intra_class_similarity=None
                        if mean_val_can_score_margin is not None
                        else mean_val_intra_class_similarity,
                        inter_class_similarity=None
                        if mean_val_can_score_margin is not None
                        else mean_val_inter_class_similarity,
                        similarity_margin=None
                        if mean_val_can_score_margin is not None
                        else mean_val_similarity_margin,
                        can_true_class_score=mean_val_can_true_class_score,
                        can_best_other_score=mean_val_can_best_other_score,
                        can_score_margin=mean_val_can_score_margin,
                        checkpoint_metric=validation_checkpoint.metric,
                        checkpoint_value=checkpoint_value,
                        checkpoint_is_best=checkpoint_improved,
                    )
                    validation_extra_metrics = {
                        "task_loss": mean_val_task_loss,
                        "precision": mean_val_precision,
                        "recall": mean_val_recall,
                        "f1": mean_val_f1,
                    }
                    if mean_val_can_score_margin is not None:
                        validation_extra_metrics.update(
                            {
                                "can_local_loss": mean_val_can_local_loss,
                                "can_true_class_score": mean_val_can_true_class_score,
                                "can_best_other_score": mean_val_can_best_other_score,
                                "can_score_margin": mean_val_can_score_margin,
                            }
                        )
                    else:
                        validation_extra_metrics.update(
                            {
                                "contrastive_loss": mean_val_contrastive_loss,
                                "triplet_loss": mean_val_triplet_loss,
                                "can_local_loss": mean_val_can_local_loss,
                                "can_global_loss": mean_val_can_global_loss,
                                "intra_class_similarity": mean_val_intra_class_similarity,
                                "inter_class_similarity": mean_val_inter_class_similarity,
                                "similarity_margin": mean_val_similarity_margin,
                            }
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
                        extra_metrics=validation_extra_metrics,
                        log_every=eval_log_every,
                    )
                    if mean_val_can_score_margin is not None:
                        self.logger.info(
                            f"[Fold {fold + 1}/{num_subjects}] "
                            f"[Epoch {epoch + 1}/{num_epochs}] "
                            f"[Validation @ train_batch {processed_batches}] "
                            f"[train_task {processed_tasks}/{tasks_per_epoch}] "
                            f"mean_loss={mean_val_loss:.4f}, "
                            f"mean_task_loss={mean_val_task_loss:.4f}, "
                            f"mean_accuracy={mean_val_acc:.4f}, "
                            f"mean_f1={mean_val_f1:.4f}, "
                            f"can_local_loss={mean_val_can_local_loss:.4f}, "
                            f"can_true_class_score={mean_val_can_true_class_score:.4f}, "
                            f"can_best_other_score={mean_val_can_best_other_score:.4f}, "
                            f"can_score_margin={mean_val_can_score_margin:.4f}, "
                            f"checkpoint_metric={validation_checkpoint.metric}, "
                            f"checkpoint_value={checkpoint_value:.4f}, "
                            f"checkpoint_is_best={checkpoint_improved}, "
                            f"validation_elapsed={self._format_seconds(time.perf_counter() - validation_start)}"
                        )
                    else:
                        self.logger.info(
                            f"[Fold {fold + 1}/{num_subjects}] "
                            f"[Epoch {epoch + 1}/{num_epochs}] "
                            f"[Validation @ train_batch {processed_batches}] "
                            f"[train_task {processed_tasks}/{tasks_per_epoch}] "
                            f"mean_loss={mean_val_loss:.4f}, "
                            f"mean_task_loss={mean_val_task_loss:.4f}, "
                            f"mean_accuracy={mean_val_acc:.4f}, "
                            f"mean_f1={mean_val_f1:.4f}, "
                            f"contrastive_loss={mean_val_contrastive_loss:.4f}, "
                            f"triplet_loss={mean_val_triplet_loss:.4f}, "
                            f"can_local_loss={mean_val_can_local_loss:.4f}, "
                            f"can_global_loss={mean_val_can_global_loss:.4f}, "
                            f"intra_class_similarity={mean_val_intra_class_similarity:.4f}, "
                            f"inter_class_similarity={mean_val_inter_class_similarity:.4f}, "
                            f"similarity_margin={mean_val_similarity_margin:.4f}, "
                            f"checkpoint_metric={validation_checkpoint.metric}, "
                            f"checkpoint_value={checkpoint_value:.4f}, "
                            f"checkpoint_is_best={checkpoint_improved}, "
                            f"validation_elapsed={self._format_seconds(time.perf_counter() - validation_start)}"
                        )

                avg_train_loss = np.mean(epoch_train_losses)
                avg_train_acc = np.mean(epoch_train_accs)
                avg_val_loss = (
                    float(np.mean(epoch_val_losses))
                    if epoch_val_losses
                    else float("nan")
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

                if (
                    self.logging_verbosity >= 1
                    and fold_summary_reference["train"] is not None
                ):
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

            checkpoint_summary = validation_checkpoint.summary()
            restored_checkpoint = validation_checkpoint.restore(
                model_variables=self.model.weights,
                optimizer_variables=self.optimizer.variables,
            )
            cv_results["validation_checkpoint_values"].append(
                checkpoint_summary["value"]
            )
            cv_results["validation_checkpoint_epochs"].append(
                checkpoint_summary["epoch"]
            )
            cv_results["validation_checkpoint_steps"].append(
                checkpoint_summary["step"]
            )
            cv_results["validation_checkpoint_metrics"].append(
                checkpoint_summary["metrics"]
            )
            if restored_checkpoint:
                self.logger.info(
                    f"[Fold {fold + 1}/{num_subjects}] "
                    "Restored best validation checkpoint for held-out evaluation: "
                    f"metric={checkpoint_summary['metric']} "
                    f"mode={checkpoint_summary['resolved_mode']} "
                    f"value={float(checkpoint_summary['value']):.4f} "
                    f"epoch={checkpoint_summary['epoch']} "
                    f"step={checkpoint_summary['step']}"
                )
            else:
                self.logger.info(
                    f"[Fold {fold + 1}/{num_subjects}] "
                    "No validation checkpoint was captured; using final training state "
                    "for held-out evaluation."
                )

            if self.config.can_support_mode == "learned_prototype_memory":
                prototype_epochs = max(0, int(self.config.prototype_finetune_epochs))
                prototype_tasks_per_epoch = (
                    self._resolve_prototype_finetune_tasks_per_epoch(train_sampler)
                )
                if prototype_epochs > 0:
                    explicit_budget = (
                        self.config.prototype_finetune_tasks_per_epoch is not None
                    )
                    budget_source = "explicit" if explicit_budget else "active_subjects"
                    self.logger.info(
                        f"[Fold {fold + 1}/{num_subjects}] "
                        "Starting learned-prototype phase-2 fine-tuning: "
                        f"epochs={prototype_epochs}, tasks_per_epoch={prototype_tasks_per_epoch}, "
                        f"budget_source={budget_source}, "
                        f"active_train_subjects={len(train_sampler.active_subjects_array)}, "
                        f"slots_per_class={self.config.learned_prototype_slots_per_class}. "
                        "Each phase-2 task is a full query-only subject task, so it is "
                        "much larger than a normal k/q episode."
                    )
                    if (
                        prototype_tasks_per_epoch
                        > max(200, 4 * len(train_sampler.active_subjects_array))
                    ):
                        self.logger.warning(
                            f"[Fold {fold + 1}/{num_subjects}] "
                            "Prototype phase-2 update budget is high for all-query "
                            f"tasks ({prototype_tasks_per_epoch} updates/epoch). "
                            "Consider lowering --prototype-finetune-tasks-per-epoch."
                        )
                for prototype_epoch in range(prototype_epochs):
                    epoch_start_time = time.perf_counter()
                    last_log_time = epoch_start_time
                    phase_losses = []
                    phase_task_losses = []
                    phase_accs = []
                    phase_can_local_losses = []
                    log_every = max(
                        1,
                        min(
                            int(getattr(self.config, "train_log_every", 10)),
                            max(1, prototype_tasks_per_epoch // 5),
                        ),
                    )
                    self.logger.info(
                        f"[Fold {fold + 1}/{num_subjects}] "
                        f"[Prototype phase {prototype_epoch + 1}/{prototype_epochs}] "
                        f"Starting {prototype_tasks_per_epoch} full-query updates"
                    )
                    for prototype_step in range(prototype_tasks_per_epoch):
                        step_start_time = time.perf_counter()
                        sampled_subject = int(
                            train_sampler.rng.choice(train_sampler.active_subjects_array)
                        )
                        prototype_task = self.dataset.build_all_query_task(
                            sampled_subject,
                            split=train_sampler.data_split,
                            use_base_index=True,
                            normalize_with_query_subject_stats=True,
                        )
                        (
                            phase_loss,
                            phase_task_loss,
                            phase_acc,
                            _phase_contrastive_loss,
                            _phase_triplet_loss,
                            phase_can_local_loss,
                            _phase_can_global_loss,
                        ) = self._train_prototype_memory_batch_step_tensors(
                            support_x_batch=tf.convert_to_tensor(
                                prototype_task["support_X"][tf.newaxis, ...],
                                dtype=tf.float32,
                            ),
                            support_y_batch=tf.convert_to_tensor(
                                prototype_task["support_y"][tf.newaxis, ...],
                                dtype=tf.int32,
                            ),
                            query_x_batch=tf.convert_to_tensor(
                                prototype_task["query_X"][tf.newaxis, ...],
                                dtype=tf.float32,
                            ),
                            query_y_batch=tf.convert_to_tensor(
                                prototype_task["query_y"][tf.newaxis, ...],
                                dtype=tf.int32,
                            ),
                        )
                        phase_losses.append(float(phase_loss))
                        phase_task_losses.append(float(phase_task_loss))
                        phase_accs.append(float(phase_acc))
                        phase_can_local_losses.append(float(phase_can_local_loss))
                        completed_steps = prototype_step + 1
                        should_log_step = (
                            completed_steps == 1
                            or completed_steps == prototype_tasks_per_epoch
                            or completed_steps % log_every == 0
                            or time.perf_counter() - last_log_time >= 60.0
                        )
                        if should_log_step:
                            elapsed = time.perf_counter() - epoch_start_time
                            seconds_per_update = elapsed / max(1, completed_steps)
                            remaining = prototype_tasks_per_epoch - completed_steps
                            eta_seconds = seconds_per_update * remaining
                            step_seconds = time.perf_counter() - step_start_time
                            query_count = int(prototype_task["query_X"].shape[0])
                            self.logger.info(
                                f"[Fold {fold + 1}/{num_subjects}] "
                                f"[Prototype phase {prototype_epoch + 1}/{prototype_epochs}] "
                                f"update {completed_steps}/{prototype_tasks_per_epoch}: "
                                f"subject={sampled_subject}, query_windows={query_count}, "
                                f"step_seconds={step_seconds:.2f}, "
                                f"elapsed={elapsed / 60.0:.1f}m, "
                                f"eta={eta_seconds / 60.0:.1f}m, "
                                f"loss={float(np.mean(phase_losses)):.4f}, "
                                f"task_loss={float(np.mean(phase_task_losses)):.4f}, "
                                f"acc={float(np.mean(phase_accs)):.4f}, "
                                f"can_local={float(np.mean(phase_can_local_losses)):.4f}"
                            )
                            last_log_time = time.perf_counter()
                    if phase_losses:
                        epoch_elapsed = time.perf_counter() - epoch_start_time
                        self.logger.info(
                            f"[Fold {fold + 1}/{num_subjects}] "
                            f"[Prototype phase {prototype_epoch + 1}/{prototype_epochs}] "
                            f"loss={float(np.mean(phase_losses)):.4f}, "
                            f"task_loss={float(np.mean(phase_task_losses)):.4f}, "
                            f"accuracy={float(np.mean(phase_accs)):.4f}, "
                            f"can_local={float(np.mean(phase_can_local_losses)):.4f}, "
                            f"elapsed_seconds={epoch_elapsed:.2f}, "
                            f"seconds_per_update={epoch_elapsed / max(1, len(phase_losses)):.2f}"
                        )

            # Held-out evaluation sweep across fixed and additional support/query sizes.
            pre_adaptation_weights = self.model.get_weights()
            pre_adaptation_optimizer_variables = self.engine.snapshot_variables(
                self.optimizer.variables
            )
            sweep_metrics_by_size = {}
            run_adaptation = k_shot_adaptation_steps > 0
            heldout_pairs_to_evaluate = heldout_eval_pairs
            if self.config.can_support_mode == "learned_prototype_memory":
                run_adaptation = False
                fixed_size_key = f"k{configured_eval_pair[0]}_q{configured_eval_pair[1]}"
                query_task = self.dataset.build_all_query_task(
                    int(test_subject),
                    split=test_sampler.data_split,
                    use_base_index=True,
                    normalize_with_query_subject_stats=True,
                )
                zero_shot_loss, zero_shot_metrics = (
                    self.evaluator.evaluate_prototype_memory_task_metrics(query_task)
                )
                k_shot_loss = float(zero_shot_loss)
                k_shot_metrics = dict(zero_shot_metrics)
                sweep_metrics_by_size[fixed_size_key] = {
                    "zero_shot_loss": zero_shot_loss,
                    "zero_shot_metrics": zero_shot_metrics,
                    "adaptation_mean_loss": 0.0,
                    "k_shot_loss": k_shot_loss,
                    "k_shot_metrics": k_shot_metrics,
                }
                size_results = cv_results["heldout_eval_by_task_size"][fixed_size_key]
                size_results["zero_shot_losses"].append(zero_shot_loss)
                size_results["zero_shot_accuracies"].append(
                    zero_shot_metrics["accuracy"]
                )
                size_results["zero_shot_precisions"].append(
                    zero_shot_metrics["precision"]
                )
                size_results["zero_shot_recalls"].append(zero_shot_metrics["recall"])
                size_results["zero_shot_f1s"].append(zero_shot_metrics["f1"])
                self._append_evaluation_diagnostics(
                    size_results, "zero_shot", zero_shot_metrics
                )
                if "transductive_accuracy" in zero_shot_metrics:
                    size_results["zero_shot_transductive_losses"].append(
                        zero_shot_metrics["transductive_loss"]
                    )
                    size_results["zero_shot_transductive_accuracies"].append(
                        zero_shot_metrics["transductive_accuracy"]
                    )
                    size_results["zero_shot_transductive_precisions"].append(
                        zero_shot_metrics["transductive_precision"]
                    )
                    size_results["zero_shot_transductive_recalls"].append(
                        zero_shot_metrics["transductive_recall"]
                    )
                    size_results["zero_shot_transductive_f1s"].append(
                        zero_shot_metrics["transductive_f1"]
                    )
                size_results["k_shot_losses"].append(k_shot_loss)
                size_results["k_shot_accuracies"].append(k_shot_metrics["accuracy"])
                size_results["k_shot_precisions"].append(k_shot_metrics["precision"])
                size_results["k_shot_recalls"].append(k_shot_metrics["recall"])
                size_results["k_shot_f1s"].append(k_shot_metrics["f1"])
                self._append_evaluation_diagnostics(
                    size_results, "k_shot", k_shot_metrics
                )
                if "transductive_accuracy" in k_shot_metrics:
                    size_results["k_shot_transductive_losses"].append(
                        k_shot_metrics["transductive_loss"]
                    )
                    size_results["k_shot_transductive_accuracies"].append(
                        k_shot_metrics["transductive_accuracy"]
                    )
                    size_results["k_shot_transductive_precisions"].append(
                        k_shot_metrics["transductive_precision"]
                    )
                    size_results["k_shot_transductive_recalls"].append(
                        k_shot_metrics["transductive_recall"]
                    )
                    size_results["k_shot_transductive_f1s"].append(
                        k_shot_metrics["transductive_f1"]
                    )
                self.logger.info(
                    f"[Fold {fold + 1}/{num_subjects}] "
                    "Prototype-only holdout evaluated on all query samples: "
                    f"queries={len(query_task['query_y'])}, "
                    f"accuracy={zero_shot_metrics['accuracy']:.4f}"
                )
                heldout_pairs_to_evaluate = []
            if not run_adaptation:
                self.logger.info(
                    f"[Fold {fold + 1}/{num_subjects}] "
                    "Skipping held-out adaptation sweep because "
                    "k_shot_adaptation_steps=0."
                )
            for eval_k_shot, eval_q_query in heldout_pairs_to_evaluate:
                size_key = f"k{eval_k_shot}_q{eval_q_query}"

                self.model.set_weights(pre_adaptation_weights)
                self.engine.restore_variable_snapshot(
                    self.optimizer.variables,
                    pre_adaptation_optimizer_variables,
                    label="pre-heldout optimizer variables",
                )
                try:
                    zero_shot_loss, zero_shot_metrics = (
                        self._evaluate_sampler_loss_and_metrics_at_task_size(
                            test_sampler,
                            num_tasks=heldout_eval_tasks,
                            k_shot=eval_k_shot,
                            q_query=eval_q_query,
                            forward_batch_size=self.train_batch_size,
                        )
                    )
                except ValueError as exc:
                    if (eval_k_shot, eval_q_query) == configured_eval_pair:
                        raise
                    self.logger.info(
                        f"[Fold {fold + 1}/{num_subjects}] "
                        f"Skipping optional held-out size {size_key}: {exc}"
                    )
                    continue
                csv_writer.write_event(
                    fold_idx=fold + 1,
                    test_subject=test_subject,
                    event_type=f"zero_shot_summary_{size_key}",
                    loss=zero_shot_loss,
                    task_loss=zero_shot_metrics["task_loss"],
                    contrastive_loss=None
                    if "can_score_margin" in zero_shot_metrics
                    else zero_shot_metrics["contrastive_loss"],
                    triplet_loss=None
                    if "can_score_margin" in zero_shot_metrics
                    else zero_shot_metrics["triplet_loss"],
                    can_local_loss=zero_shot_metrics["can_local_loss"],
                    can_global_loss=None
                    if "can_score_margin" in zero_shot_metrics
                    else zero_shot_metrics["can_global_loss"],
                    accuracy=zero_shot_metrics["accuracy"],
                    precision=zero_shot_metrics["precision"],
                    recall=zero_shot_metrics["recall"],
                    f1=zero_shot_metrics["f1"],
                    intra_class_similarity=None
                    if "can_score_margin" in zero_shot_metrics
                    else zero_shot_metrics["intra_class_similarity"],
                    inter_class_similarity=None
                    if "can_score_margin" in zero_shot_metrics
                    else zero_shot_metrics["inter_class_similarity"],
                    can_true_class_score=zero_shot_metrics.get(
                        "can_true_class_score"
                    ),
                    can_best_other_score=zero_shot_metrics.get(
                        "can_best_other_score"
                    ),
                    can_score_margin=zero_shot_metrics.get("can_score_margin"),
                )
                if (eval_k_shot, eval_q_query) == configured_eval_pair:
                    if k_shot_adaptation_steps > 0:
                        progress.log_adaptation_start(
                            fold_idx=fold + 1,
                            total_folds=num_subjects,
                            test_subject=int(test_subject),
                            adaptation_steps=k_shot_adaptation_steps,
                        )

                if run_adaptation:
                    adaptation_losses = self._adapt_on_sampler_at_task_size(
                        test_sampler,
                        adaptation_steps=k_shot_adaptation_steps,
                        k_shot=eval_k_shot,
                        q_query=eval_q_query,
                    )
                    csv_writer.write_event(
                        fold_idx=fold + 1,
                        test_subject=test_subject,
                        event_type=f"adaptation_phase_{size_key}",
                        loss=float(np.mean(adaptation_losses))
                        if adaptation_losses
                        else 0.0,
                    )
                    k_shot_loss, k_shot_metrics = (
                        self._evaluate_sampler_loss_and_metrics_at_task_size(
                            test_sampler,
                            num_tasks=heldout_eval_tasks,
                            k_shot=eval_k_shot,
                            q_query=eval_q_query,
                            forward_batch_size=self.train_batch_size,
                        )
                    )
                else:
                    adaptation_losses = []
                    # With zero adaptation steps, k-shot reflects the zero-shot state.
                    k_shot_loss = float(zero_shot_loss)
                    k_shot_metrics = dict(zero_shot_metrics)

                csv_writer.write_event(
                    fold_idx=fold + 1,
                    test_subject=test_subject,
                    event_type=f"k_shot_summary_{size_key}",
                    loss=k_shot_loss,
                    task_loss=k_shot_metrics["task_loss"],
                    contrastive_loss=None
                    if "can_score_margin" in k_shot_metrics
                    else k_shot_metrics["contrastive_loss"],
                    triplet_loss=None
                    if "can_score_margin" in k_shot_metrics
                    else k_shot_metrics["triplet_loss"],
                    can_local_loss=k_shot_metrics["can_local_loss"],
                    can_global_loss=None
                    if "can_score_margin" in k_shot_metrics
                    else k_shot_metrics["can_global_loss"],
                    accuracy=k_shot_metrics["accuracy"],
                    precision=k_shot_metrics["precision"],
                    recall=k_shot_metrics["recall"],
                    f1=k_shot_metrics["f1"],
                    intra_class_similarity=None
                    if "can_score_margin" in k_shot_metrics
                    else k_shot_metrics["intra_class_similarity"],
                    inter_class_similarity=None
                    if "can_score_margin" in k_shot_metrics
                    else k_shot_metrics["inter_class_similarity"],
                    can_true_class_score=k_shot_metrics.get("can_true_class_score"),
                    can_best_other_score=k_shot_metrics.get("can_best_other_score"),
                    can_score_margin=k_shot_metrics.get("can_score_margin"),
                )

                sweep_metrics_by_size[size_key] = {
                    "zero_shot_loss": zero_shot_loss,
                    "zero_shot_metrics": zero_shot_metrics,
                    "adaptation_mean_loss": (
                        float(np.mean(adaptation_losses)) if adaptation_losses else 0.0
                    ),
                    "k_shot_loss": k_shot_loss,
                    "k_shot_metrics": k_shot_metrics,
                }
                size_results = cv_results["heldout_eval_by_task_size"][size_key]
                size_results["zero_shot_losses"].append(zero_shot_loss)
                size_results["zero_shot_accuracies"].append(
                    zero_shot_metrics["accuracy"]
                )
                size_results["zero_shot_precisions"].append(
                    zero_shot_metrics["precision"]
                )
                size_results["zero_shot_recalls"].append(zero_shot_metrics["recall"])
                size_results["zero_shot_f1s"].append(zero_shot_metrics["f1"])
                self._append_evaluation_diagnostics(
                    size_results, "zero_shot", zero_shot_metrics
                )
                if "transductive_accuracy" in zero_shot_metrics:
                    size_results["zero_shot_transductive_losses"].append(
                        zero_shot_metrics["transductive_loss"]
                    )
                    size_results["zero_shot_transductive_accuracies"].append(
                        zero_shot_metrics["transductive_accuracy"]
                    )
                    size_results["zero_shot_transductive_precisions"].append(
                        zero_shot_metrics["transductive_precision"]
                    )
                    size_results["zero_shot_transductive_recalls"].append(
                        zero_shot_metrics["transductive_recall"]
                    )
                    size_results["zero_shot_transductive_f1s"].append(
                        zero_shot_metrics["transductive_f1"]
                    )
                size_results["k_shot_losses"].append(k_shot_loss)
                size_results["k_shot_accuracies"].append(k_shot_metrics["accuracy"])
                size_results["k_shot_precisions"].append(k_shot_metrics["precision"])
                size_results["k_shot_recalls"].append(k_shot_metrics["recall"])
                size_results["k_shot_f1s"].append(k_shot_metrics["f1"])
                self._append_evaluation_diagnostics(
                    size_results, "k_shot", k_shot_metrics
                )
                if "transductive_accuracy" in k_shot_metrics:
                    size_results["k_shot_transductive_losses"].append(
                        k_shot_metrics["transductive_loss"]
                    )
                    size_results["k_shot_transductive_accuracies"].append(
                        k_shot_metrics["transductive_accuracy"]
                    )
                    size_results["k_shot_transductive_precisions"].append(
                        k_shot_metrics["transductive_precision"]
                    )
                    size_results["k_shot_transductive_recalls"].append(
                        k_shot_metrics["transductive_recall"]
                    )
                    size_results["k_shot_transductive_f1s"].append(
                        k_shot_metrics["transductive_f1"]
                    )

                self.logger.info(
                    f"[Fold {fold + 1}/{num_subjects}] [Heldout size {size_key}] "
                    f"zero_shot_acc={zero_shot_metrics['accuracy']:.4f}, "
                    f"k_shot_acc={k_shot_metrics['accuracy']:.4f}"
                )

            fixed_size_key = f"k{configured_eval_pair[0]}_q{configured_eval_pair[1]}"
            fixed_size_metrics = sweep_metrics_by_size[fixed_size_key]
            zero_shot_loss = fixed_size_metrics["zero_shot_loss"]
            zero_shot_metrics = fixed_size_metrics["zero_shot_metrics"]
            fixed_adaptation_mean_loss = fixed_size_metrics["adaptation_mean_loss"]
            k_shot_loss = fixed_size_metrics["k_shot_loss"]
            k_shot_metrics = fixed_size_metrics["k_shot_metrics"]

            # Keep legacy fixed-size event names for downstream tooling compatibility.
            csv_writer.write_event(
                fold_idx=fold + 1,
                test_subject=test_subject,
                event_type="zero_shot_summary",
                loss=zero_shot_loss,
                task_loss=zero_shot_metrics["task_loss"],
                contrastive_loss=None
                if "can_score_margin" in zero_shot_metrics
                else zero_shot_metrics["contrastive_loss"],
                triplet_loss=None
                if "can_score_margin" in zero_shot_metrics
                else zero_shot_metrics["triplet_loss"],
                can_local_loss=zero_shot_metrics["can_local_loss"],
                can_global_loss=None
                if "can_score_margin" in zero_shot_metrics
                else zero_shot_metrics["can_global_loss"],
                accuracy=zero_shot_metrics["accuracy"],
                precision=zero_shot_metrics["precision"],
                recall=zero_shot_metrics["recall"],
                f1=zero_shot_metrics["f1"],
                intra_class_similarity=None
                if "can_score_margin" in zero_shot_metrics
                else zero_shot_metrics["intra_class_similarity"],
                inter_class_similarity=None
                if "can_score_margin" in zero_shot_metrics
                else zero_shot_metrics["inter_class_similarity"],
                can_true_class_score=zero_shot_metrics.get("can_true_class_score"),
                can_best_other_score=zero_shot_metrics.get("can_best_other_score"),
                can_score_margin=zero_shot_metrics.get("can_score_margin"),
            )
            if run_adaptation:
                csv_writer.write_event(
                    fold_idx=fold + 1,
                    test_subject=test_subject,
                    event_type="adaptation_phase",
                    loss=fixed_adaptation_mean_loss,
                )
            csv_writer.write_event(
                fold_idx=fold + 1,
                test_subject=test_subject,
                event_type="k_shot_summary",
                loss=k_shot_loss,
                task_loss=k_shot_metrics["task_loss"],
                contrastive_loss=None
                if "can_score_margin" in k_shot_metrics
                else k_shot_metrics["contrastive_loss"],
                triplet_loss=None
                if "can_score_margin" in k_shot_metrics
                else k_shot_metrics["triplet_loss"],
                can_local_loss=k_shot_metrics["can_local_loss"],
                can_global_loss=None
                if "can_score_margin" in k_shot_metrics
                else k_shot_metrics["can_global_loss"],
                accuracy=k_shot_metrics["accuracy"],
                precision=k_shot_metrics["precision"],
                recall=k_shot_metrics["recall"],
                f1=k_shot_metrics["f1"],
                intra_class_similarity=None
                if "can_score_margin" in k_shot_metrics
                else k_shot_metrics["intra_class_similarity"],
                inter_class_similarity=None
                if "can_score_margin" in k_shot_metrics
                else k_shot_metrics["inter_class_similarity"],
                can_true_class_score=k_shot_metrics.get("can_true_class_score"),
                can_best_other_score=k_shot_metrics.get("can_best_other_score"),
                can_score_margin=k_shot_metrics.get("can_score_margin"),
            )

            progress.log_subject_summary(
                stage="Zero-shot",
                fold_idx=fold + 1,
                total_folds=num_subjects,
                test_subject=int(test_subject),
                loss=zero_shot_loss,
                metrics=zero_shot_metrics,
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
            self._append_evaluation_diagnostics(
                cv_results, "zero_shot", zero_shot_metrics
            )
            if "transductive_accuracy" in zero_shot_metrics:
                cv_results["zero_shot_transductive_losses"].append(
                    zero_shot_metrics["transductive_loss"]
                )
                cv_results["zero_shot_transductive_accuracies"].append(
                    zero_shot_metrics["transductive_accuracy"]
                )
                cv_results["zero_shot_transductive_precisions"].append(
                    zero_shot_metrics["transductive_precision"]
                )
                cv_results["zero_shot_transductive_recalls"].append(
                    zero_shot_metrics["transductive_recall"]
                )
                cv_results["zero_shot_transductive_f1s"].append(
                    zero_shot_metrics["transductive_f1"]
                )
            cv_results["k_shot_losses"].append(k_shot_loss)
            cv_results["k_shot_accuracies"].append(k_shot_metrics["accuracy"])
            cv_results["k_shot_precisions"].append(k_shot_metrics["precision"])
            cv_results["k_shot_recalls"].append(k_shot_metrics["recall"])
            cv_results["k_shot_f1s"].append(k_shot_metrics["f1"])
            self._append_evaluation_diagnostics(
                cv_results, "k_shot", k_shot_metrics
            )
            if "transductive_accuracy" in k_shot_metrics:
                cv_results["k_shot_transductive_losses"].append(
                    k_shot_metrics["transductive_loss"]
                )
                cv_results["k_shot_transductive_accuracies"].append(
                    k_shot_metrics["transductive_accuracy"]
                )
                cv_results["k_shot_transductive_precisions"].append(
                    k_shot_metrics["transductive_precision"]
                )
                cv_results["k_shot_transductive_recalls"].append(
                    k_shot_metrics["transductive_recall"]
                )
                cv_results["k_shot_transductive_f1s"].append(
                    k_shot_metrics["transductive_f1"]
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
            completed_folds = fold + 1
            should_log_checkpoint_summary = (
                completed_folds < num_subjects
                and completed_folds % fold_checkpoint_interval == 0
            )
            if should_log_checkpoint_summary:
                completion_pct = 100.0 * completed_folds / max(1, num_subjects)
                self._log_cross_validation_aggregate(
                    cv_results,
                    title=(
                        "CROSS-VALIDATION RESULTS "
                        f"({completed_folds}/{num_subjects} folds, {completion_pct:.1f}% complete)"
                    ),
                )

        self._log_cross_validation_aggregate(
            cv_results,
            title="CROSS-VALIDATION RESULTS",
        )

        return cv_results
