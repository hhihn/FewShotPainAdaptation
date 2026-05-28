import csv
import copy
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from data_loaders.pain_meta_dataset import PainMetaDataset
from data_loaders.loso_cross_validator import LOSOCrossValidator
from data_loaders.pain_ds_config import PainDatasetConfig
from utils.logger import setup_logger
from utils.reproducibility import set_global_reproducibility
from utils.training_progress import TrainingProgressReporter
from learner.cross_validation_results import CrossValidationResultRecorder
from learner.episodic_learning_engine import EpisodicLearningEngine
from learner.episode_evaluation_service import EpisodeEvaluationService
from learner.heldout_adaptation_service import HeldoutAdaptationService
from learner.loso_training_runner import LosoTrainingRunner
from learner.model_architecture_writer import ModelArchitectureWriter
from learner.task_batch_pipeline import TaskBatchPipeline
from learner.validation_checkpoint import ValidationCheckpointTracker


class FewShotPainLearner:
    """Train and evaluate few-shot pain adaptation models.

    The learner is the public facade that wires together data loading, LOSO
    orchestration, TensorFlow execution, evaluation, and reporting services.
    """

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
        self.can_margin_loss_weight = float(config.can_margin_loss_weight)
        self.can_margin_target = float(config.can_margin_target)
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
        self._compiled_prototype_memory_batch_step = None
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
            "can_margin_loss_weight": self.config.can_margin_loss_weight,
            "can_margin_target": self.config.can_margin_target,
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
            "export_can_feature_maps": self.config.export_can_feature_maps,
            "export_raw_can_feature_maps": self.config.export_raw_can_feature_maps,
            "embedding_dim": self.embedding_dim,
            "encoder_backend": self.config.encoder_backend,
            "eegnet_temporal_filters": self.config.eegnet_temporal_filters,
            "eegnet_depth_multiplier": self.config.eegnet_depth_multiplier,
            "eegnet_separable_filters": self.config.eegnet_separable_filters,
            "eegnet_temporal_kernel_size": self.config.eegnet_temporal_kernel_size,
            "eegnet_separable_kernel_size": self.config.eegnet_separable_kernel_size,
            "eegnet_pool_size_1": self.config.eegnet_pool_size_1,
            "eegnet_pool_size_2": self.config.eegnet_pool_size_2,
            "eegnet_dropout_rate": self.config.eegnet_dropout_rate,
            "eegnet_l2_weight": self.config.eegnet_l2_weight,
            "crossmod_num_heads": self.config.crossmod_num_heads,
            "crossmod_hidden_dim": self.config.crossmod_hidden_dim,
            "crossmod_num_layers": self.config.crossmod_num_layers,
            "crossmod_positional_base": self.config.crossmod_positional_base,
            "crossmod_attention_dropout_rate": self.config.crossmod_attention_dropout_rate,
            "crossmod_ff_activation": self.config.crossmod_ff_activation,
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
        if self.config.encoder_backend == "crossmod":
            self.logger.info("Encoder: CrossMod EDA/ECG feature-map encoder")
        else:
            self.logger.info("Encoder: EEGNet-style joint sensor encoder")
        self.logger.info(
            f"Logging verbosity={self.logging_verbosity} (0=minimal, 1=standard, 2=detailed)"
        )

    def _augment_training_inputs(self, x: tf.Tensor) -> tf.Tensor:
        """Apply training-only signal augmentation.

        Args:
            x: Input tensor to augment.
        """
        return self.engine.augment_training_inputs(x)

    def _release_model_resources(self, clear_session: bool = True) -> None:
        """Release model and optimizer resources.

        Args:
            clear_session: Whether to clear Keras backend state.
        """
        self.engine.release_model_resources(clear_session=clear_session)

    def _rebuild_model(self, clear_session: bool = True) -> None:
        """Build a fresh model and optimizer.

        Args:
            clear_session: Whether to clear stale TensorFlow graph state first.
        """
        self.engine.rebuild_model(clear_session=clear_session)

    def _reset_model_state_for_new_fold(self) -> None:
        """Restore initial model and optimizer state for a new fold.

        Compiled functions are reused after variables are restored.
        """
        self.engine.reset_model_state_for_new_fold()

    def _build_compiled_train_batch_step(self) -> None:
        """Build the compiled train-step function.

        The implementation is delegated to the episodic learning engine.
        """
        self.engine.build_compiled_train_batch_step()

    def _build_compiled_eval_batch_step(self) -> None:
        """Build the compiled evaluation function.

        The implementation is delegated to the episodic learning engine.
        """
        self.engine.build_compiled_eval_batch_step()

    def _build_compiled_prototype_memory_batch_step(self) -> None:
        """Build the compiled prototype-memory update function.

        The implementation is delegated to the episodic learning engine.
        """
        self.engine.build_compiled_prototype_memory_batch_step()

    def _get_loso_fold_subjects(self) -> list[int]:
        """Return held-out subjects selected by LOSO configuration.

        Single-fold and start/stop index settings are resolved here.
        """
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
        """Evaluate a batch of tasks without optimizer updates.

        This facade delegates to the engine's compiled implementation.
        """
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
        """Run a compiled optimizer update over one task batch.

        This facade delegates to the engine's compiled implementation.
        """
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
        """Run one learned-prototype-memory optimizer update.

        The update path is used for CAN phase-2 fine-tuning.
        """
        return self.engine.train_prototype_memory_batch_step_tensors(
            support_x_batch=support_x_batch,
            support_y_batch=support_y_batch,
            query_x_batch=query_x_batch,
            query_y_batch=query_y_batch,
        )

    def _resolve_prototype_finetune_tasks_per_epoch(self, train_sampler) -> int:
        """Return the phase-2 learned-prototype update budget.

        Args:
            train_sampler: Training sampler used to infer a default budget.
        """
        configured_tasks = self.config.prototype_finetune_tasks_per_epoch
        if configured_tasks is not None:
            return max(1, int(configured_tasks))
        active_subject_count = int(len(train_sampler.active_subjects_array))
        return max(1, active_subject_count)

    def _iter_prototype_finetune_task_batches(
        self,
        train_sampler,
        prototype_updates_per_epoch: int,
    ):
        """Yield episodic task batches for phase-2 updates.

        Args:
            train_sampler: Training sampler used to draw tasks.
            prototype_updates_per_epoch: Number of prototype updates per epoch.
        """
        total_sampled_tasks = (
            max(1, int(prototype_updates_per_epoch)) * self.train_batch_size
        )
        yield from self._iter_prefetched_task_batches(
            train_sampler,
            total_sampled_tasks,
        )

    def _compute_model_aux_loss(self, dtype: tf.dtypes.DType) -> tf.Tensor:
        """Return model auxiliary losses.

        Args:
            dtype: Output dtype for the summed auxiliary loss.
        """
        return self.engine.compute_model_aux_loss(dtype)

    def _apply_gradients(self, loss: tf.Tensor, tape: tf.GradientTape) -> tf.Tensor:
        """Apply gradients for the current model update.

        Args:
            loss: Scalar objective tensor.
            tape: Active gradient tape.
        """
        return self.engine.apply_gradients(loss, tape)

    def _compute_batch_all_triplet_loss(
        self, embeddings: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        """Compute batch-all triplet loss.

        Args:
            embeddings: Embedding tensor.
            labels: Labels aligned with embeddings.
        """
        return self.engine.compute_batch_all_triplet_loss(embeddings, labels)

    def _compute_batch_hard_triplet_loss(
        self, embeddings: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        """Compute batch-hard triplet loss.

        Args:
            embeddings: Embedding tensor.
            labels: Labels aligned with embeddings.
        """
        return self.engine.compute_batch_hard_triplet_loss(embeddings, labels)

    def _compute_triplet_center_loss(
        self, embeddings: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        """Compute triplet-center loss.

        Args:
            embeddings: Embedding tensor.
            labels: Labels aligned with embeddings.
        """
        return self.engine.compute_triplet_center_loss(embeddings, labels)

    def _compute_triplet_loss(
        self, embeddings: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        """Dispatch the configured triplet mining strategy.

        Args:
            embeddings: Embedding tensor.
            labels: Labels aligned with embeddings.
        """
        return self.engine.compute_triplet_loss(embeddings, labels)

    def _compute_task_batch_objective(
        self,
        episode_outputs: dict[str, tf.Tensor],
        support_y_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        """Compute objective tensors for batched episode outputs.

        Args:
            episode_outputs: Task-major model output dictionary.
            support_y_batch: Batched support labels.
            query_y_batch: Batched query labels.
        """
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
        """Run one task and compute losses.

        Args:
            support_x: Support windows.
            support_y: Support labels.
            query_x: Query windows.
            query_y: Query labels.
            training: Whether child layers run in training mode.
            return_similarity_scores: Whether to include similarity scores.
        """
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
        """Run multiple tasks with batched encoding and losses.

        Args:
            support_x_batch: Task-major support windows.
            support_y_batch: Task-major support labels.
            query_x_batch: Task-major query windows.
            query_y_batch: Task-major query labels.
            training: Whether child layers run in training mode.
            return_similarity_scores: Whether to include similarity scores.
        """
        return self.engine.forward_task_batch(
            support_x_batch=support_x_batch,
            support_y_batch=support_y_batch,
            query_x_batch=query_x_batch,
            query_y_batch=query_y_batch,
            training=training,
            return_similarity_scores=return_similarity_scores,
        )

    def train_step(self, support_x, support_y, query_x, query_y):
        """Run one training step on a single task.

        Args:
            support_x: Support windows.
            support_y: Support labels.
            query_x: Query windows.
            query_y: Query labels.
        """
        return self.engine.train_step(support_x, support_y, query_x, query_y)

    @staticmethod
    def _stack_task_batch_numpy(
        task_batch: list[dict],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pack task dictionaries into dense NumPy arrays.

        Args:
            task_batch: List of task dictionaries with uniform shapes.
        """
        return TaskBatchPipeline.stack_task_batch_numpy(task_batch)

    @staticmethod
    def _stack_task_batch(
        task_batch: list[dict],
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Pack task dictionaries into dense TensorFlow tensors.

        Args:
            task_batch: List of task dictionaries with uniform shapes.
        """
        return TaskBatchPipeline.stack_task_batch(task_batch)

    @staticmethod
    def _sample_and_stack_task_batch_numpy(
        sampler,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample one task batch and pack it into NumPy arrays.

        Args:
            sampler: Episodic sampler exposing ``get_task``.
            batch_size: Number of tasks to sample.
        """
        return TaskBatchPipeline.sample_and_stack_task_batch_numpy(
            sampler,
            batch_size,
        )

    def _iter_prefetched_task_batches(
        self,
        sampler,
        tasks_per_epoch: int,
    ):
        """Yield stacked NumPy task batches with optional prefetch.

        Args:
            sampler: Episodic sampler exposing ``get_task``.
            tasks_per_epoch: Number of tasks to sample.
        """
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
        """Yield task tensor chunks sized by embedding batch size.

        Args:
            support_x_batch: Task-major support windows.
            support_y_batch: Task-major support labels.
            query_x_batch: Task-major query windows.
            query_y_batch: Task-major query labels.
        """
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
        """Apply training augmentation to one task chunk.

        Args:
            support_x_chunk: Task-major support windows.
            query_x_chunk: Task-major query windows.
        """
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
        """Forward one eager chunk and normalize outputs.

        Args:
            support_x_chunk: Task-major support windows.
            support_y_chunk: Task-major support labels.
            query_x_chunk: Task-major query windows.
            query_y_chunk: Task-major query labels.
            training: Whether child layers run in training mode.
            return_similarity_scores: Whether to include similarity scores.
        """
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
        """Return the mean over tensors collected from chunks.

        Args:
            tensor_parts: List of tensors to concatenate before reducing.
        """
        return tf.reduce_mean(tf.concat(tensor_parts, axis=0))

    @staticmethod
    def _train_metric_tensors_from_chunk_outputs(
        chunk_outputs: dict[str, tf.Tensor],
        query_y_chunk: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        """Return train metric tensors for one normalized chunk.

        Args:
            chunk_outputs: Task-major output dictionary.
            query_y_chunk: Task-major query labels.
        """
        return EpisodicLearningEngine.train_metric_tensors_from_chunk_outputs(
            chunk_outputs,
            query_y_chunk,
        )

    def _split_batched_similarity_scores(
        self,
        similarity_scores: tf.Tensor,
        query_y_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Split batched scores into true-class and other-class groups.

        Args:
            similarity_scores: Task-major query-by-class scores.
            query_y_batch: Task-major query labels.
        """
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
        """Return flattened eval losses, labels, predictions, and scores.

        Args:
            chunk_outputs: Task-major output dictionary.
            query_y_chunk: Task-major query labels.
        """
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
        """Run eager optimizer update for one task batch.

        Args:
            support_x_batch: Task-major support windows.
            support_y_batch: Task-major support labels.
            query_x_batch: Task-major query windows.
            query_y_batch: Task-major query labels.
        """
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
        """Run compiled train step with eager fallback.

        Args:
            support_x_batch: Task-major support windows.
            support_y_batch: Task-major support labels.
            query_x_batch: Task-major query windows.
            query_y_batch: Task-major query labels.
        """
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
        """Run eager task-batch evaluation.

        Args:
            support_x_batch: Task-major support windows.
            support_y_batch: Task-major support labels.
            query_x_batch: Task-major query windows.
            query_y_batch: Task-major query labels.
        """
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
        """Run compiled task-batch evaluation with eager fallback.

        Args:
            support_x_batch: Task-major support windows.
            support_y_batch: Task-major support labels.
            query_x_batch: Task-major query windows.
            query_y_batch: Task-major query labels.
        """
        return self.engine.eval_task_batch_step_tensors(
            support_x_batch=support_x_batch,
            support_y_batch=support_y_batch,
            query_x_batch=query_x_batch,
            query_y_batch=query_y_batch,
        )

    @staticmethod
    def _task_batch_has_uniform_shapes(task_batch: list[dict]) -> bool:
        """Return whether support/query tensors have uniform shapes.

        Args:
            task_batch: List of task dictionaries to inspect.
        """
        return TaskBatchPipeline.task_batch_has_uniform_shapes(task_batch)

    def train_batch_step(self, task_batch: list[dict]) -> tuple[tf.Tensor, ...]:
        """Run one optimizer update using a task batch.

        Args:
            task_batch: List of task dictionaries.
        """
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
        """Evaluate one task without updating weights.

        Args:
            support_x: Support windows.
            support_y: Support labels.
            query_x: Query windows.
            query_y: Query labels.
        """
        return self.engine.evaluate_task(support_x, support_y, query_x, query_y)

    def evaluate_batch_step(
        self, task_batch: list[dict]
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Evaluate a batch of tasks without updating weights.

        Args:
            task_batch: List of task dictionaries.
        """
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
        """Split similarities into true-class and other-class groups.

        Args:
            similarity_scores: Query-by-class similarity matrix.
            y_true: True query labels.
        """
        return EpisodeEvaluationService.split_similarity_scores(
            similarity_scores,
            y_true,
        )

    @staticmethod
    def _compute_similarity_metrics(
        intra_class_scores: np.ndarray, inter_class_scores: np.ndarray
    ) -> dict:
        """Aggregate intra/inter-class similarity statistics.

        Args:
            intra_class_scores: Similarities assigned to true classes.
            inter_class_scores: Similarities assigned to other classes.
        """
        return EpisodeEvaluationService.compute_similarity_metrics(
            intra_class_scores,
            inter_class_scores,
        )

    def _evaluate_task_batch_loss_and_metrics(
        self,
        task_batch: list[dict],
        *,
        forward_batch_size: int | None = None,
        can_support_mode: str | None = None,
    ) -> tuple[float, dict]:
        """Evaluate a task batch and aggregate metrics.

        Args:
            task_batch: List of task dictionaries.
            forward_batch_size: Optional tasks per batched forward pass.
            can_support_mode: Optional temporary CAN support mode override.
        """
        return self.evaluator.evaluate_task_batch_loss_and_metrics(
            task_batch,
            forward_batch_size=forward_batch_size,
            can_support_mode=can_support_mode,
        )

    def _compute_macro_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute accuracy and macro precision/recall/F1.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
        """
        return self.evaluator.compute_macro_metrics(y_true, y_pred)

    def _evaluate_sampler_loss_and_metrics(
        self,
        sampler,
        num_tasks: int,
        *,
        forward_batch_size: int | None = None,
    ) -> tuple[float, dict]:
        """Evaluate sampled tasks and aggregate metrics.

        Args:
            sampler: Episodic sampler exposing ``get_task``.
            num_tasks: Number of tasks to sample.
            forward_batch_size: Optional tasks per batched forward pass.
        """
        return self.evaluator.evaluate_sampler_loss_and_metrics(
            sampler,
            num_tasks,
            forward_batch_size=forward_batch_size,
        )

    @staticmethod
    def _set_sampler_task_size(sampler, k_shot: int, q_query: int) -> None:
        """Update sampler task size in place.

        Args:
            sampler: Episodic sampler with mutable k/q fields.
            k_shot: Support samples per class.
            q_query: Query samples per class.
        """
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
        """Evaluate sampler metrics with temporary task size.

        Args:
            sampler: Episodic sampler with mutable k/q fields.
            num_tasks: Number of tasks to sample.
            k_shot: Temporary support samples per class.
            q_query: Temporary query samples per class.
            forward_batch_size: Optional tasks per batched forward pass.
        """
        return self.evaluator.evaluate_sampler_loss_and_metrics_at_task_size(
            sampler,
            num_tasks=num_tasks,
            k_shot=k_shot,
            q_query=q_query,
            forward_batch_size=forward_batch_size,
        )

    def _sample_tasks_at_task_size(
        self,
        sampler,
        *,
        num_tasks: int,
        k_shot: int,
        q_query: int,
    ) -> list[dict]:
        """Sample held-out tasks with temporary task size.

        Args:
            sampler: Episodic sampler with mutable k/q fields.
            num_tasks: Number of tasks to sample.
            k_shot: Temporary support samples per class.
            q_query: Temporary query samples per class.
        """
        original_k = int(sampler.k_shot)
        original_q = int(sampler.q_query)
        self._set_sampler_task_size(sampler, k_shot=k_shot, q_query=q_query)
        try:
            return [sampler.get_task() for _ in range(num_tasks)]
        finally:
            self._set_sampler_task_size(
                sampler,
                k_shot=original_k,
                q_query=original_q,
            )

    def _evaluate_learned_prototype_bank_reference(
        self,
        *,
        fold: int,
        num_subjects: int,
        test_subject: int,
        test_sampler,
        label: str,
    ) -> tuple[dict, float, dict]:
        """Evaluate all held-out queries with learned prototype-memory support."""
        query_task = self.dataset.build_all_query_task(
            int(test_subject),
            split=test_sampler.data_split,
            use_base_index=True,
            normalize_with_query_subject_stats=True,
        )
        loss, metrics = self.evaluator.evaluate_prototype_memory_task_metrics(
            query_task
        )
        self.logger.info(
            f"[Fold {fold + 1}/{num_subjects}] "
            f"{label} learned-prototype bank evaluated on all query samples: "
            f"queries={len(query_task['query_y'])}, "
            f"accuracy={metrics['accuracy']:.4f}"
        )
        return query_task, loss, metrics

    def _write_phase2_initial_prototype_bank_evaluation(
        self,
        *,
        fold: int,
        num_subjects: int,
        test_subject: int,
        test_sampler,
        result_recorder,
    ) -> tuple[dict, float, dict]:
        """Write pre-update learned-prototype bank performance to progress CSV."""
        query_task, loss, metrics = self._evaluate_learned_prototype_bank_reference(
            fold=fold,
            num_subjects=num_subjects,
            test_subject=test_subject,
            test_sampler=test_sampler,
            label="Phase-2 initial",
        )
        result_recorder.write_metric_event(
            fold_idx=fold + 1,
            test_subject=test_subject,
            event_type="prototype_bank_phase2_initial_summary",
            loss=loss,
            metrics=metrics,
        )
        return query_task, loss, metrics

    def _write_phase2_initial_sampled_support_evaluation(
        self,
        *,
        fold: int,
        num_subjects: int,
        test_subject: int,
        test_sampler,
        configured_eval_pair: tuple[int, int],
        heldout_eval_tasks: int,
        result_recorder,
    ) -> tuple[float, dict]:
        """Write pre-update sampled-support performance to progress CSV.

        The sampler RNG is restored after this audit evaluation so adding the
        phase-2 initial row does not change later hold-out evaluation draws.
        """
        eval_k_shot, eval_q_query = configured_eval_pair
        rng_state = None
        rng = getattr(test_sampler, "rng", None)
        bit_generator = getattr(rng, "bit_generator", None)
        if bit_generator is not None:
            rng_state = copy.deepcopy(bit_generator.state)

        try:
            support_task_batch = self._sample_tasks_at_task_size(
                test_sampler,
                num_tasks=heldout_eval_tasks,
                k_shot=eval_k_shot,
                q_query=eval_q_query,
            )
            loss, metrics = self._evaluate_task_batch_loss_and_metrics(
                support_task_batch,
                forward_batch_size=self.train_batch_size,
                can_support_mode="sampled",
            )
        finally:
            if rng_state is not None:
                bit_generator.state = rng_state

        result_recorder.write_metric_event(
            fold_idx=fold + 1,
            test_subject=test_subject,
            event_type=(
                "support_samples_phase2_initial_summary_"
                f"k{eval_k_shot}_q{eval_q_query}"
            ),
            loss=loss,
            metrics=metrics,
        )
        self.logger.info(
            f"[Fold {fold + 1}/{num_subjects}] "
            "Phase-2 initial sampled-support hold-out evaluation: "
            f"k={eval_k_shot}, q={eval_q_query}, tasks={heldout_eval_tasks}, "
            f"accuracy={metrics['accuracy']:.4f}"
        )
        return loss, metrics

    def _evaluate_learned_prototype_holdout_sweep(
        self,
        *,
        fold: int,
        num_subjects: int,
        test_subject: int,
        test_sampler,
        heldout_eval_pairs: list[tuple[int, int]],
        configured_eval_pair: tuple[int, int],
        heldout_eval_tasks: int,
        result_recorder,
    ) -> dict:
        """Evaluate learned-prototype zero-shot and sampled-support holdout sizes."""
        query_task, zero_shot_loss, zero_shot_metrics = (
            self._evaluate_learned_prototype_bank_reference(
                fold=fold,
                num_subjects=num_subjects,
                test_subject=test_subject,
                test_sampler=test_sampler,
                label="Post-phase-2",
            )
        )

        sweep_metrics_by_size = {}
        for eval_k_shot, eval_q_query in heldout_eval_pairs:
            size_key = f"k{eval_k_shot}_q{eval_q_query}"
            try:
                k_shot_task_batch = self._sample_tasks_at_task_size(
                    test_sampler,
                    num_tasks=heldout_eval_tasks,
                    k_shot=eval_k_shot,
                    q_query=eval_q_query,
                )
                k_shot_loss, k_shot_metrics = (
                    self._evaluate_task_batch_loss_and_metrics(
                        k_shot_task_batch,
                        forward_batch_size=self.train_batch_size,
                        can_support_mode="sampled",
                    )
                )
            except ValueError as exc:
                if (eval_k_shot, eval_q_query) == configured_eval_pair:
                    raise
                self.logger.info(
                    f"[Fold {fold + 1}/{num_subjects}] "
                    f"Skipping optional learned-prototype held-out size "
                    f"{size_key}: {exc}"
                )
                continue

            result_recorder.write_metric_event(
                fold_idx=fold + 1,
                test_subject=test_subject,
                event_type=f"k_shot_summary_{size_key}",
                loss=k_shot_loss,
                metrics=k_shot_metrics,
            )
            sweep_metrics_by_size[size_key] = (
                result_recorder.record_heldout_size_result(
                    size_key=size_key,
                    zero_shot_loss=zero_shot_loss,
                    zero_shot_metrics=zero_shot_metrics,
                    adaptation_losses=[],
                    k_shot_loss=k_shot_loss,
                    k_shot_metrics=k_shot_metrics,
                    zero_shot_task_batch=[query_task],
                    k_shot_task_batch=k_shot_task_batch,
                )
            )
            self.logger.info(
                f"[Fold {fold + 1}/{num_subjects}] "
                f"[Learned prototype heldout size {size_key}] "
                f"zero_shot_acc={zero_shot_metrics['accuracy']:.4f}, "
                f"support_conditioned_acc={k_shot_metrics['accuracy']:.4f}"
            )
        return sweep_metrics_by_size

    def _adapt_on_sampler_at_task_size(
        self,
        sampler,
        *,
        adaptation_steps: int,
        k_shot: int,
        q_query: int,
    ) -> list[float]:
        """Run adaptation using temporary task size.

        Args:
            sampler: Held-out episodic sampler.
            adaptation_steps: Number of adaptation updates.
            k_shot: Temporary support samples per class.
            q_query: Temporary query samples per class.
        """
        return self.adaptation_service.adapt_on_sampler_at_task_size(
            sampler,
            adaptation_steps=adaptation_steps,
            k_shot=k_shot,
            q_query=q_query,
        )

    def _save_model_architecture(self, sample_task: dict, output_path: str) -> str:
        """Build the model and save architecture summaries.

        Args:
            sample_task: Task dictionary used to build model variables.
            output_path: Output path for the text summary.
        """
        return self.architecture_writer.save_model_architecture(
            sample_task,
            output_path,
        )

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        """Format elapsed seconds as a compact human-readable string.

        Args:
            seconds: Duration in seconds.
        """
        seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m {secs}s"
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    def _write_can_alignment_summary(
        self,
        *,
        progress_file: str,
        fold_idx: int,
        test_subject: int,
        k_shot: int,
        q_query: int,
        zero_shot_metrics: dict,
        k_shot_metrics: dict,
        zero_shot_support_mode: str | None = None,
        k_shot_support_mode: str | None = None,
    ) -> str | None:
        """Write per-fold CAN alignment summary CSV.

        Args:
            progress_file: Fold progress CSV path used to derive output path.
            fold_idx: One-based fold index.
            test_subject: Held-out subject identifier.
            k_shot: Support samples per class.
            q_query: Query samples per class.
            zero_shot_metrics: Zero-shot evaluation metrics.
            k_shot_metrics: Post-adaptation evaluation metrics.
        """
        if getattr(self.config, "attention_mode", "none") != "can":
            return None
        if "can_mean_alignment" not in zero_shot_metrics:
            return None

        progress_path = Path(progress_file)
        output_path = progress_path.with_name(
            progress_path.name.replace(
                "_training_progress.csv",
                "_can_alignment_summary.csv",
            )
        )
        fieldnames = [
            "fold",
            "test_subject",
            "phase",
            "k_shot",
            "q_query",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "can_mean_alignment",
            "can_true_class_score",
            "can_best_other_score",
            "can_score_margin",
            "can_support_mode",
        ]
        rows = [
            ("zero_shot", zero_shot_metrics),
            ("k_shot", k_shot_metrics),
        ]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for phase, metrics in rows:
                writer.writerow(
                    {
                        "fold": fold_idx,
                        "test_subject": int(test_subject),
                        "phase": phase,
                        "k_shot": int(k_shot),
                        "q_query": int(q_query),
                        "accuracy": metrics.get("accuracy"),
                        "precision": metrics.get("precision"),
                        "recall": metrics.get("recall"),
                        "f1": metrics.get("f1"),
                        "can_mean_alignment": metrics.get("can_mean_alignment"),
                        "can_true_class_score": metrics.get("can_true_class_score"),
                        "can_best_other_score": metrics.get("can_best_other_score"),
                        "can_score_margin": metrics.get("can_score_margin"),
                        "can_support_mode": (
                            zero_shot_support_mode
                            if phase == "zero_shot"
                            else k_shot_support_mode
                        )
                        or getattr(self.config, "can_support_mode", "sampled"),
                    }
                )
        return str(output_path)

    def _write_can_sample_statistics(
        self,
        *,
        progress_file: str,
        fold_idx: int,
        test_subject: int,
        k_shot: int,
        q_query: int,
        zero_shot_task_batch: list[dict],
        k_shot_task_batch: list[dict],
        zero_shot_support_mode: str | None = None,
        k_shot_support_mode: str | None = None,
    ) -> str | None:
        """Write per-sample CAN diagnostic statistics.

        Args:
            progress_file: Fold progress CSV path used to derive output path.
            fold_idx: One-based fold index.
            test_subject: Held-out subject identifier.
            k_shot: Support samples per class.
            q_query: Query samples per class.
            zero_shot_task_batch: Tasks evaluated before adaptation.
            k_shot_task_batch: Tasks evaluated after adaptation.
        """
        if getattr(self.config, "attention_mode", "none") != "can":
            return None

        rows = []
        rows.extend(
            self.evaluator.collect_can_sample_statistics(
                zero_shot_task_batch,
                phase="zero_shot",
                can_support_mode=zero_shot_support_mode,
            )
        )
        rows.extend(
            self.evaluator.collect_can_sample_statistics(
                k_shot_task_batch,
                phase="k_shot",
                can_support_mode=k_shot_support_mode,
            )
        )
        if not rows:
            return None

        progress_path = Path(progress_file)
        output_path = progress_path.with_name(
            progress_path.name.replace(
                "_training_progress.csv",
                "_can_sample_statistics.csv",
            )
        )
        class_fields = []
        for class_index in range(int(self.config.n_way)):
            class_fields.extend(
                [f"logit_class_{class_index}", f"can_score_class_{class_index}"]
            )
        fieldnames = [
            "fold",
            "test_subject",
            "phase",
            "k_shot",
            "q_query",
            "task_index",
            "sample_index",
            "true_label",
            "pred_label",
            "correct",
            "loss",
            "can_mean_alignment",
            "can_true_class_score",
            "can_best_other_score",
            "can_score_margin",
            *class_fields,
        ]
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "fold": fold_idx,
                        "test_subject": int(test_subject),
                        "k_shot": int(k_shot),
                        "q_query": int(q_query),
                        **row,
                    }
                )
        return str(output_path)

    def _write_can_feature_export(
        self,
        *,
        progress_file: str,
        fold_idx: int,
        test_subject: int,
        k_shot: int,
        q_query: int,
        zero_shot_task_batch: list[dict],
        k_shot_task_batch: list[dict],
        zero_shot_support_mode: str | None = None,
        k_shot_support_mode: str | None = None,
    ) -> str | None:
        """Write compact CAN feature-map exports for subject-adaptation analysis."""
        if getattr(self.config, "attention_mode", "none") != "can":
            return None
        if not bool(getattr(self.config, "export_can_feature_maps", True)):
            return None

        include_raw = bool(getattr(self.config, "export_raw_can_feature_maps", False))
        phase_exports = {
            "zero_shot": self.evaluator.collect_can_feature_export(
                zero_shot_task_batch,
                phase="zero_shot",
                can_support_mode=zero_shot_support_mode,
                include_raw_feature_maps=include_raw,
            ),
            "k_shot": self.evaluator.collect_can_feature_export(
                k_shot_task_batch,
                phase="k_shot",
                can_support_mode=k_shot_support_mode,
                include_raw_feature_maps=include_raw,
            ),
        }
        if not any(phase_exports.values()):
            return None

        payload = {
            "fold": np.array(int(fold_idx), dtype=np.int32),
            "test_subject": np.array(int(test_subject), dtype=np.int32),
            "k_shot": np.array(int(k_shot), dtype=np.int32),
            "q_query": np.array(int(q_query), dtype=np.int32),
            "include_raw_feature_maps": np.array(include_raw),
        }
        for phase, export in phase_exports.items():
            for key, value in export.items():
                payload[f"{phase}_{key}"] = value

        progress_path = Path(progress_file)
        output_path = progress_path.with_name(
            progress_path.name.replace(
                "_training_progress.csv",
                "_can_feature_exports.npz",
            )
        )
        np.savez_compressed(output_path, **payload)
        return str(output_path)

    def train(
        self,
        training_progress_output_dir: str = "outputs/training_progress",
        save_model_architecture_first_run: bool = True,
        model_architecture_output_path: str = "outputs/model_architecture/model_summary.txt",
    ):
        """Train with leave-one-subject-out cross-validation.

        Args:
            training_progress_output_dir: Directory for progress CSV files.
            save_model_architecture_first_run: Whether to save architecture once.
            model_architecture_output_path: Output path for architecture summary.
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
        """Run the internal LOSO training workflow.

        Args:
            training_progress_output_dir: Directory for progress CSV files.
            save_model_architecture_first_run: Whether to save architecture once.
            model_architecture_output_path: Output path for architecture summary.
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

        result_recorder = CrossValidationResultRecorder(
            heldout_eval_pairs=heldout_eval_pairs,
            training_progress_output_dir=training_progress_output_dir,
            csv_flush_every_events=self.config.csv_flush_every_events,
            validation_checkpoint_metric=self.config.validation_checkpoint_metric,
            validation_checkpoint_mode=self.config.validation_checkpoint_mode,
            logger=self.logger,
        )
        cv_results = result_recorder.results

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
        architecture_saved = False

        for fold, test_subject in enumerate(fold_subjects):
            fold_start_time = time.perf_counter()
            progress.log_fold_start(
                fold_idx=fold + 1, total_folds=num_subjects, test_subject=test_subject
            )
            progress_file = result_recorder.start_fold(
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
                result_recorder.set_model_architecture_file(architecture_path)
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
                        can_margin_loss,
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
                        result_recorder.write_train_update(
                            fold_idx=fold + 1,
                            test_subject=test_subject,
                            epoch=epoch + 1,
                            epoch_total=num_epochs,
                            step=processed_tasks,
                            step_total=tasks_per_epoch,
                            loss=float(loss),
                            task_loss=float(task_loss),
                            contrastive_loss=None
                            if can_mode
                            else float(contrastive_loss),
                            triplet_loss=None if can_mode else float(triplet_loss),
                            can_local_loss=float(can_local_loss),
                            can_global_loss=None
                            if can_mode
                            else float(can_global_loss),
                            accuracy=float(acc),
                            can_margin_loss=float(can_margin_loss)
                            if can_mode
                            else None,
                        )
                    train_extra_metrics = {
                        "task_loss": float(task_loss),
                        "can_local_loss": float(can_local_loss),
                    }
                    if self.attention_mode == "can":
                        train_extra_metrics["can_margin_loss"] = float(can_margin_loss)
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
                                f"can_margin_loss={float(can_margin_loss):.4f}, "
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
                    validation_can_margin_losses = []
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
                        validation_can_margin_losses.append(
                            val_metrics["can_margin_loss"]
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
                    mean_val_can_margin_loss = float(
                        np.mean(validation_can_margin_losses)
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
                        "can_margin_loss": mean_val_can_margin_loss,
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
                    result_recorder.write_validation_step(
                        fold_idx=fold + 1,
                        test_subject=test_subject,
                        epoch=epoch + 1,
                        epoch_total=num_epochs,
                        step=processed_tasks,
                        step_total=tasks_per_epoch,
                        loss=mean_val_loss,
                        metrics=validation_checkpoint_metrics,
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
                                "can_margin_loss": mean_val_can_margin_loss,
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
                            f"can_margin_loss={mean_val_can_margin_loss:.4f}, "
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
                    result_recorder.log_composite_summary(
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
            result_recorder.record_validation_checkpoint(checkpoint_summary)
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
                prototype_updates_per_epoch = (
                    self._resolve_prototype_finetune_tasks_per_epoch(train_sampler)
                )
                if prototype_epochs > 0:
                    explicit_budget = (
                        self.config.prototype_finetune_tasks_per_epoch is not None
                    )
                    budget_source = "explicit" if explicit_budget else "active_subjects"
                    support_samples_per_task = int(
                        self.config.n_way * self.config.k_shot
                    )
                    query_samples_per_task = int(
                        self.config.n_way * self.config.q_query
                    )
                    self.logger.info(
                        f"[Fold {fold + 1}/{num_subjects}] "
                        "Starting learned-prototype phase-2 fine-tuning: "
                        f"epochs={prototype_epochs}, updates_per_epoch={prototype_updates_per_epoch}, "
                        f"budget_source={budget_source}, "
                        f"active_train_subjects={len(train_sampler.active_subjects_array)}, "
                        f"batch_size={self.train_batch_size}, "
                        f"support_samples_per_task={support_samples_per_task}, "
                        f"query_samples_per_task={query_samples_per_task}, "
                        f"slots_per_class={self.config.learned_prototype_slots_per_class}. "
                        "Each phase-2 update uses one configured batched episodic task batch; "
                        "sampled support tensors are carried through the batch interface while "
                        "learned prototype memory supplies CAN support."
                    )
                    self._write_phase2_initial_sampled_support_evaluation(
                        fold=fold,
                        num_subjects=num_subjects,
                        test_subject=test_subject,
                        test_sampler=test_sampler,
                        configured_eval_pair=configured_eval_pair,
                        heldout_eval_tasks=heldout_eval_tasks,
                        result_recorder=result_recorder,
                    )
                    self._write_phase2_initial_prototype_bank_evaluation(
                        fold=fold,
                        num_subjects=num_subjects,
                        test_subject=test_subject,
                        test_sampler=test_sampler,
                        result_recorder=result_recorder,
                    )
                for prototype_epoch in range(prototype_epochs):
                    epoch_start_time = time.perf_counter()
                    last_log_time = epoch_start_time
                    phase_loss_sum = tf.constant(0.0, dtype=tf.float32)
                    phase_task_loss_sum = tf.constant(0.0, dtype=tf.float32)
                    phase_acc_sum = tf.constant(0.0, dtype=tf.float32)
                    phase_can_local_loss_sum = tf.constant(0.0, dtype=tf.float32)
                    phase_can_margin_loss_sum = tf.constant(0.0, dtype=tf.float32)
                    phase_update_count = 0
                    log_every = max(
                        1,
                        min(
                            int(getattr(self.config, "train_log_every", 10)),
                            max(1, prototype_updates_per_epoch // 5),
                        ),
                    )
                    self.logger.info(
                        f"[Fold {fold + 1}/{num_subjects}] "
                        f"[Prototype phase {prototype_epoch + 1}/{prototype_epochs}] "
                        "Starting "
                        f"{prototype_updates_per_epoch} batched episodic "
                        "prototype-bank updates"
                    )
                    for prototype_step, (
                        current_batch_size,
                        (
                            support_x_np,
                            support_y_np,
                            query_x_np,
                            query_y_np,
                        ),
                    ) in enumerate(
                        self._iter_prototype_finetune_task_batches(
                            train_sampler,
                            prototype_updates_per_epoch,
                        )
                    ):
                        step_start_time = time.perf_counter()
                        (
                            phase_loss,
                            phase_task_loss,
                            phase_acc,
                            _phase_contrastive_loss,
                            _phase_triplet_loss,
                            phase_can_local_loss,
                            _phase_can_global_loss,
                            phase_can_margin_loss,
                        ) = self._train_prototype_memory_batch_step_tensors(
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
                        completed_steps = prototype_step + 1
                        phase_update_count = completed_steps
                        phase_loss_sum += tf.cast(phase_loss, tf.float32)
                        phase_task_loss_sum += tf.cast(phase_task_loss, tf.float32)
                        phase_acc_sum += tf.cast(phase_acc, tf.float32)
                        phase_can_local_loss_sum += tf.cast(
                            phase_can_local_loss,
                            tf.float32,
                        )
                        phase_can_margin_loss_sum += tf.cast(
                            phase_can_margin_loss,
                            tf.float32,
                        )
                        should_log_step = (
                            completed_steps == 1
                            or completed_steps == prototype_updates_per_epoch
                            or completed_steps % log_every == 0
                            or time.perf_counter() - last_log_time >= 60.0
                        )
                        if should_log_step:
                            elapsed = time.perf_counter() - epoch_start_time
                            seconds_per_update = elapsed / max(1, completed_steps)
                            remaining = prototype_updates_per_epoch - completed_steps
                            eta_seconds = seconds_per_update * remaining
                            step_seconds = time.perf_counter() - step_start_time
                            support_count = int(support_x_np.shape[1])
                            query_count = int(query_x_np.shape[1])
                            mean_denominator = tf.cast(
                                completed_steps,
                                tf.float32,
                            )
                            mean_phase_loss = float(phase_loss_sum / mean_denominator)
                            mean_phase_task_loss = float(
                                phase_task_loss_sum / mean_denominator
                            )
                            mean_phase_acc = float(phase_acc_sum / mean_denominator)
                            mean_phase_can_local_loss = float(
                                phase_can_local_loss_sum / mean_denominator
                            )
                            mean_phase_can_margin_loss = float(
                                phase_can_margin_loss_sum / mean_denominator
                            )
                            self.logger.info(
                                f"[Fold {fold + 1}/{num_subjects}] "
                                f"[Prototype phase {prototype_epoch + 1}/{prototype_epochs}] "
                                f"update {completed_steps}/{prototype_updates_per_epoch}: "
                                f"batch_size={current_batch_size}, "
                                f"support_samples_per_task={support_count}, "
                                f"query_samples_per_task={query_count}, "
                                f"step_seconds={step_seconds:.2f}, "
                                f"elapsed={elapsed / 60.0:.1f}m, "
                                f"eta={eta_seconds / 60.0:.1f}m, "
                                f"loss={mean_phase_loss:.4f}, "
                                f"task_loss={mean_phase_task_loss:.4f}, "
                                f"acc={mean_phase_acc:.4f}, "
                                f"can_local={mean_phase_can_local_loss:.4f}, "
                                f"can_margin={mean_phase_can_margin_loss:.4f}"
                            )
                            last_log_time = time.perf_counter()
                    if phase_update_count > 0:
                        epoch_elapsed = time.perf_counter() - epoch_start_time
                        mean_denominator = tf.cast(phase_update_count, tf.float32)
                        mean_phase_loss = float(phase_loss_sum / mean_denominator)
                        mean_phase_task_loss = float(
                            phase_task_loss_sum / mean_denominator
                        )
                        mean_phase_acc = float(phase_acc_sum / mean_denominator)
                        mean_phase_can_local_loss = float(
                            phase_can_local_loss_sum / mean_denominator
                        )
                        mean_phase_can_margin_loss = float(
                            phase_can_margin_loss_sum / mean_denominator
                        )
                        self.logger.info(
                            f"[Fold {fold + 1}/{num_subjects}] "
                            f"[Prototype phase {prototype_epoch + 1}/{prototype_epochs}] "
                            f"loss={mean_phase_loss:.4f}, "
                            f"task_loss={mean_phase_task_loss:.4f}, "
                            f"accuracy={mean_phase_acc:.4f}, "
                            f"can_local={mean_phase_can_local_loss:.4f}, "
                            f"can_margin={mean_phase_can_margin_loss:.4f}, "
                            f"elapsed_seconds={epoch_elapsed:.2f}, "
                            f"seconds_per_update={epoch_elapsed / max(1, phase_update_count):.2f}"
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
                sweep_metrics_by_size = (
                    self._evaluate_learned_prototype_holdout_sweep(
                        fold=fold,
                        num_subjects=num_subjects,
                        test_subject=test_subject,
                        test_sampler=test_sampler,
                        heldout_eval_pairs=heldout_eval_pairs,
                        configured_eval_pair=configured_eval_pair,
                        heldout_eval_tasks=heldout_eval_tasks,
                        result_recorder=result_recorder,
                    )
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
                    zero_shot_task_batch = self._sample_tasks_at_task_size(
                        test_sampler,
                        num_tasks=heldout_eval_tasks,
                        k_shot=eval_k_shot,
                        q_query=eval_q_query,
                    )
                    zero_shot_loss, zero_shot_metrics = (
                        self._evaluate_task_batch_loss_and_metrics(
                            zero_shot_task_batch,
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
                result_recorder.write_metric_event(
                    fold_idx=fold + 1,
                    test_subject=test_subject,
                    event_type=f"zero_shot_summary_{size_key}",
                    loss=zero_shot_loss,
                    metrics=zero_shot_metrics,
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
                    result_recorder.write_adaptation_event(
                        fold_idx=fold + 1,
                        test_subject=test_subject,
                        event_type=f"adaptation_phase_{size_key}",
                        adaptation_losses=adaptation_losses,
                    )
                    k_shot_task_batch = self._sample_tasks_at_task_size(
                        test_sampler,
                        num_tasks=heldout_eval_tasks,
                        k_shot=eval_k_shot,
                        q_query=eval_q_query,
                    )
                    k_shot_loss, k_shot_metrics = (
                        self._evaluate_task_batch_loss_and_metrics(
                            k_shot_task_batch,
                            forward_batch_size=self.train_batch_size,
                        )
                    )
                else:
                    adaptation_losses = []
                    # With zero adaptation steps, k-shot reflects the zero-shot state.
                    k_shot_loss = float(zero_shot_loss)
                    k_shot_metrics = dict(zero_shot_metrics)
                    k_shot_task_batch = zero_shot_task_batch

                result_recorder.write_metric_event(
                    fold_idx=fold + 1,
                    test_subject=test_subject,
                    event_type=f"k_shot_summary_{size_key}",
                    loss=k_shot_loss,
                    metrics=k_shot_metrics,
                )

                sweep_metrics_by_size[size_key] = (
                    result_recorder.record_heldout_size_result(
                        size_key=size_key,
                        zero_shot_loss=zero_shot_loss,
                        zero_shot_metrics=zero_shot_metrics,
                        adaptation_losses=adaptation_losses,
                        k_shot_loss=k_shot_loss,
                        k_shot_metrics=k_shot_metrics,
                        zero_shot_task_batch=zero_shot_task_batch,
                        k_shot_task_batch=k_shot_task_batch,
                    )
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
            zero_shot_task_batch = fixed_size_metrics.get("zero_shot_task_batch", [])
            k_shot_task_batch = fixed_size_metrics.get("k_shot_task_batch", [])

            result_recorder.write_legacy_heldout_events(
                fold_idx=fold + 1,
                test_subject=test_subject,
                zero_shot_loss=zero_shot_loss,
                zero_shot_metrics=zero_shot_metrics,
                adaptation_mean_loss=fixed_adaptation_mean_loss,
                k_shot_loss=k_shot_loss,
                k_shot_metrics=k_shot_metrics,
                run_adaptation=run_adaptation,
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

            result_recorder.record_fold_result(
                fold_idx=fold + 1,
                test_subject=test_subject,
                fold_results=fold_results,
                zero_shot_loss=zero_shot_loss,
                zero_shot_metrics=zero_shot_metrics,
                k_shot_loss=k_shot_loss,
                k_shot_metrics=k_shot_metrics,
            )
            zero_shot_support_mode = getattr(
                self.config,
                "can_support_mode",
                "sampled",
            )
            k_shot_support_mode = zero_shot_support_mode
            if self.config.can_support_mode == "learned_prototype_memory":
                k_shot_support_mode = "sampled"
            can_alignment_file = self._write_can_alignment_summary(
                progress_file=progress_file,
                fold_idx=fold + 1,
                test_subject=int(test_subject),
                k_shot=configured_eval_pair[0],
                q_query=configured_eval_pair[1],
                zero_shot_metrics=zero_shot_metrics,
                k_shot_metrics=k_shot_metrics,
                zero_shot_support_mode=zero_shot_support_mode,
                k_shot_support_mode=k_shot_support_mode,
            )
            if can_alignment_file is not None:
                result_recorder.record_can_alignment_summary_file(can_alignment_file)
                self.logger.info(
                    f"[Fold {fold + 1}/{num_subjects}] "
                    f"Saved CAN alignment summary to {can_alignment_file}"
                )
            can_sample_statistics_file = self._write_can_sample_statistics(
                progress_file=progress_file,
                fold_idx=fold + 1,
                test_subject=int(test_subject),
                k_shot=configured_eval_pair[0],
                q_query=configured_eval_pair[1],
                zero_shot_task_batch=zero_shot_task_batch,
                k_shot_task_batch=k_shot_task_batch,
                zero_shot_support_mode=zero_shot_support_mode,
                k_shot_support_mode=k_shot_support_mode,
            )
            if can_sample_statistics_file is not None:
                result_recorder.record_can_sample_statistics_file(
                    can_sample_statistics_file
                )
                self.logger.info(
                    f"[Fold {fold + 1}/{num_subjects}] "
                    f"Saved CAN sample statistics to {can_sample_statistics_file}"
                )
            can_feature_export_file = self._write_can_feature_export(
                progress_file=progress_file,
                fold_idx=fold + 1,
                test_subject=int(test_subject),
                k_shot=configured_eval_pair[0],
                q_query=configured_eval_pair[1],
                zero_shot_task_batch=zero_shot_task_batch,
                k_shot_task_batch=k_shot_task_batch,
                zero_shot_support_mode=zero_shot_support_mode,
                k_shot_support_mode=k_shot_support_mode,
            )
            if can_feature_export_file is not None:
                result_recorder.record_can_feature_export_file(can_feature_export_file)
                self.logger.info(
                    f"[Fold {fold + 1}/{num_subjects}] "
                    f"Saved CAN feature export to {can_feature_export_file}"
                )
            result_recorder.close_fold()
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
                result_recorder.log_cross_validation_aggregate(
                    title=(
                        "CROSS-VALIDATION RESULTS "
                        f"({completed_folds}/{num_subjects} folds, {completion_pct:.1f}% complete)"
                    ),
                )

        result_recorder.log_cross_validation_aggregate(
            title="CROSS-VALIDATION RESULTS",
        )

        return cv_results
