import gc
import math

import tensorflow as tf
from tensorflow import keras

from architecture.multimodal_proto_net import MultimodalPrototypicalNetwork


class EpisodicLearningEngine:
    """Manage TensorFlow model lifecycle and episodic tensor execution.

    The engine owns model construction, optimizer state, compiled train/eval
    steps, and low-level loss computations for the learner facade.
    """

    def __init__(self, learner):
        """Initialize the engine around a learner facade.

        Args:
            learner: FewShotPainLearner instance providing config and services.
        """
        self.learner = learner
        self._compiled_train_batch_step = None
        self._compiled_eval_batch_step = None
        self._compiled_prototype_memory_batch_step = None
        self._initial_model_weights = None
        self._initial_optimizer_variables = None
        self._phase1_optimizer = None

    def __getattr__(self, name):
        """Delegate unknown attributes to the learner facade.

        Args:
            name: Attribute name requested on the engine.
        """
        return getattr(self.learner, name)

    @property
    def model(self):
        """Return the active learner model.

        The property keeps engine code synchronized with the facade state.
        """
        return self.learner.model

    @model.setter
    def model(self, value):
        """Set the active learner model.

        Args:
            value: Keras model instance or None.
        """
        self.learner.model = value

    @property
    def optimizer(self):
        """Return the active optimizer.

        The optimizer is stored on the learner facade.
        """
        return self.learner.optimizer

    @optimizer.setter
    def optimizer(self, value):
        """Set the active optimizer.

        Args:
            value: Keras optimizer instance or None.
        """
        self.learner.optimizer = value

    def augment_training_inputs(self, x: tf.Tensor) -> tf.Tensor:
        """Apply training-only signal augmentation to input windows.

        Args:
            x: Input tensor to augment.
        """
        if self.gaussian_noise_std <= 0:
            return x
        noise = keras.random.normal(
            shape=tf.shape(x),
            mean=0.0,
            stddev=tf.cast(self.gaussian_noise_std, x.dtype),
            dtype=x.dtype,
            seed=self.augmentation_seed_generator,
        )
        return x + noise

    def release_model_resources(self, clear_session: bool = True) -> None:
        """Release model, optimizer, and compiled-function references.

        Args:
            clear_session: Whether to clear the global Keras backend session.
        """
        self._compiled_train_batch_step = None
        self._compiled_eval_batch_step = None
        self._compiled_prototype_memory_batch_step = None
        self.learner._compiled_train_batch_step = None
        self.learner._compiled_eval_batch_step = None
        self.learner._compiled_prototype_memory_batch_step = None
        self._initial_model_weights = None
        self._initial_optimizer_variables = None
        self._phase1_optimizer = None
        self.model = None
        self.optimizer = None
        if clear_session:
            tf.keras.backend.clear_session()
        gc.collect()

    def rebuild_model(self, clear_session: bool = True) -> None:
        """Build a fresh model and optimizer for the current config.

        Args:
            clear_session: Whether to release existing TensorFlow graph state first.
        """
        if clear_session:
            self.release_model_resources(clear_session=True)

        self.learner.sequence_length = int(self.config.sequence_length)
        self.learner.num_sensors = int(self.config.num_sensors)

        self.model = MultimodalPrototypicalNetwork(
            sequence_length=self.config.sequence_length,
            num_sensors=self.num_sensors,
            num_classes=self.config.n_way,
            attention_mode=self.config.attention_mode,
            can_attention_temperature=self.config.can_attention_temperature,
            can_meta_hidden_dim=self.config.can_meta_hidden_dim,
            can_meta_depth=self.config.can_meta_depth,
            can_meta_activation=self.config.can_meta_activation,
            can_temporal_pooling=self.config.can_temporal_pooling,
            can_local_pool_temperature=self.config.can_local_pool_temperature,
            can_logit_scale_initial=self.config.can_logit_scale_initial,
            can_support_mode="sampled",
            learned_prototype_slots_per_class=self.config.learned_prototype_slots_per_class,
            prototype_feature_normalization=self.config.prototype_feature_normalization,
            prototype_aggregation=self.config.prototype_aggregation,
            prototype_attention_temperature=self.config.prototype_attention_temperature,
            eegnet_temporal_filters=self.config.eegnet_temporal_filters,
            eegnet_depth_multiplier=self.config.eegnet_depth_multiplier,
            eegnet_separable_filters=self.config.eegnet_separable_filters,
            eegnet_temporal_kernel_size=self.config.eegnet_temporal_kernel_size,
            eegnet_separable_kernel_size=self.config.eegnet_separable_kernel_size,
            eegnet_pool_size_1=self.config.eegnet_pool_size_1,
            eegnet_pool_size_2=self.config.eegnet_pool_size_2,
            eegnet_dropout_rate=self.config.eegnet_dropout_rate,
            eegnet_l2_weight=self.config.eegnet_l2_weight,
            eegnet_normalization=self.config.eegnet_normalization,
            eegnet_group_norm_groups=self.config.eegnet_group_norm_groups,
            encoder_backend=self.config.encoder_backend,
            crossmod_num_heads=self.config.crossmod_num_heads,
            crossmod_hidden_dim=self.config.crossmod_hidden_dim,
            crossmod_num_layers=self.config.crossmod_num_layers,
            crossmod_positional_base=self.config.crossmod_positional_base,
            crossmod_attention_dropout_rate=self.config.crossmod_attention_dropout_rate,
            crossmod_ff_activation=self.config.crossmod_ff_activation,
            crossmod_fusion_mode=self.config.crossmod_fusion_mode,
            seed=self.seed,
        )
        self.optimizer = self.build_optimizer()
        self._phase1_optimizer = self.optimizer
        self.initialize_model_and_optimizer_variables()
        self.build_compiled_train_batch_step()
        self.build_compiled_eval_batch_step()
        self.build_compiled_prototype_memory_batch_step()
        self.capture_initial_model_state()

    def build_learning_rate(
        self,
        *,
        updates_per_epoch: int | None = None,
        num_epochs: int | None = None,
    ):
        """Return the configured learning-rate object.

        Constant schedules return a scalar; cosine schedules return a Keras
        learning-rate schedule.

        Args:
            updates_per_epoch: Optional optimizer-update count per epoch. When
                omitted, this is derived from the phase-1 task budget.
            num_epochs: Optional schedule epoch count. When omitted, the
                configured phase-1 epoch count is used.
        """
        schedule_name = str(getattr(self.config, "lr_schedule", "constant")).lower()
        if schedule_name == "constant":
            return self.learning_rate
        if schedule_name == "cosine":
            if updates_per_epoch is None:
                updates_per_epoch = max(
                    1,
                    math.ceil(
                        max(1, int(self.config.tasks_per_epoch))
                        / self.train_batch_size
                    ),
                )
            else:
                updates_per_epoch = max(1, int(updates_per_epoch))
            resolved_epochs = (
                max(1, int(self.config.num_epochs))
                if num_epochs is None
                else max(1, int(num_epochs))
            )
            decay_steps = max(
                1, updates_per_epoch * resolved_epochs
            )
            return keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=decay_steps,
                alpha=float(getattr(self.config, "lr_decay_alpha", 0.1)),
            )
        raise ValueError(f"Unknown lr_schedule: {schedule_name}")

    def build_optimizer(
        self,
        *,
        updates_per_epoch: int | None = None,
        num_epochs: int | None = None,
    ) -> keras.optimizers.Optimizer:
        """Build a fresh AdamW optimizer and learning-rate schedule.

        Args:
            updates_per_epoch: Optional optimizer-update count used to size a
                cosine schedule.
            num_epochs: Optional epoch count used to size a cosine schedule.
        """
        optimizer_kwargs = {
            "learning_rate": self.build_learning_rate(
                updates_per_epoch=updates_per_epoch,
                num_epochs=num_epochs,
            ),
            "weight_decay": 1e-4,
        }
        if self.gradient_clip_norm is not None:
            optimizer_kwargs["clipnorm"] = self.gradient_clip_norm
        return keras.optimizers.AdamW(**optimizer_kwargs)

    def restart_optimizer_for_prototype_phase(
        self,
        *,
        updates_per_epoch: int,
        num_epochs: int,
    ) -> None:
        """Start phase two with fresh optimizer state and schedule.

        The phase-one optimizer is retained so the next LOSO fold can restore
        its original optimizer and schedule. The new optimizer is built against
        every model variable to remain compatible with optional held-out
        adaptation after prototype fine-tuning.
        """
        if self._phase1_optimizer is None:
            self._phase1_optimizer = self.optimizer
        self.optimizer = self.build_optimizer(
            updates_per_epoch=updates_per_epoch,
            num_epochs=num_epochs,
        )
        if hasattr(self.optimizer, "build"):
            self.optimizer.build(self.model.trainable_variables)
        self.build_compiled_prototype_memory_batch_step()

    def make_dummy_episode_tensors(
        self,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Create one dummy episode for variable initialization.

        Returns:
            Support windows, support labels, query windows, and query labels.
        """
        class_ids = tf.range(int(self.config.n_way), dtype=tf.int32)
        support_y = tf.repeat(class_ids, repeats=int(self.config.k_shot))
        query_y = tf.repeat(class_ids, repeats=int(self.config.q_query))
        support_x = tf.zeros(
            (
                self.support_size,
                self.sequence_length,
                self.num_sensors,
            ),
            dtype=tf.float32,
        )
        query_x = tf.zeros(
            (
                self.query_size,
                self.sequence_length,
                self.num_sensors,
            ),
            dtype=tf.float32,
        )
        return support_x, support_y, query_x, query_y

    def initialize_model_and_optimizer_variables(self) -> None:
        """Build model weights and optimizer slot variables without training.

        A dummy forward pass initializes model variables and optional CAN
        prototype-memory variables before optimizer slots are built.
        """
        support_x, support_y, query_x, _ = self.make_dummy_episode_tensors()
        self.model.forward_episode(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            training=False,
        )
        if getattr(self.model, "can_enabled", False):
            original_support_mode = self.model.can_support_mode
            self.model.can_support_mode = "learned_prototype_memory"
            try:
                self.model.forward_episode(
                    support_x=support_x,
                    support_y=support_y,
                    query_x=query_x,
                    training=False,
                )
            finally:
                self.model.can_support_mode = original_support_mode
        if hasattr(self.optimizer, "build"):
            self.optimizer.build(self.model.trainable_variables)

    @staticmethod
    def snapshot_variables(variables: list[tf.Variable]) -> list[tf.Tensor]:
        """Return immutable tensor snapshots for variables.

        Args:
            variables: Variables to copy.
        """
        return [tf.identity(variable) for variable in variables]

    @staticmethod
    def restore_variable_snapshot(
        variables: list[tf.Variable],
        snapshot: list[tf.Tensor],
        *,
        label: str,
    ) -> None:
        """Assign a captured snapshot back into matching variables.

        Args:
            variables: Variables to restore.
            snapshot: Tensor values captured earlier.
            label: Human-readable label for error messages.
        """
        if len(variables) != len(snapshot):
            raise RuntimeError(
                f"Cannot restore {label}: variable count changed from "
                f"{len(snapshot)} to {len(variables)}."
            )
        for variable, value in zip(variables, snapshot):
            variable.assign(value)

    def capture_initial_model_state(self) -> None:
        """Capture post-build untrained model and optimizer state.

        The captured state is reused to reset each LOSO fold without retracing.
        """
        self._initial_model_weights = self.snapshot_variables(self.model.weights)
        self._initial_optimizer_variables = self.snapshot_variables(
            self.optimizer.variables
        )

    def reset_model_state_for_new_fold(self) -> None:
        """Restore fold-start model and optimizer state.

        The restore avoids rebuilding compiled TensorFlow functions between folds.
        """
        if self._phase1_optimizer is not None and self.optimizer is not self._phase1_optimizer:
            self.optimizer = self._phase1_optimizer

        if (
            self._initial_model_weights is None
            or self._initial_optimizer_variables is None
        ):
            self.initialize_model_and_optimizer_variables()
            self.capture_initial_model_state()

        self.restore_variable_snapshot(
            self.model.weights,
            self._initial_model_weights,
            label="model weights",
        )
        self.restore_variable_snapshot(
            self.optimizer.variables,
            self._initial_optimizer_variables,
            label="optimizer variables",
        )

    def build_compiled_train_batch_step(self) -> None:
        """Build the compiled batched train-step function.

        The compiled function is also mirrored onto the learner facade for
        backwards-compatible access.
        """
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
        self.learner._compiled_train_batch_step = self._compiled_train_batch_step

    def build_compiled_eval_batch_step(self) -> None:
        """Build the compiled batched evaluation function.

        The input signature permits variable support/query counts for evaluation.
        """
        self._compiled_eval_batch_step = tf.function(
            self._eval_task_batch_step_compiled_impl,
            reduce_retracing=True,
            input_signature=[
                tf.TensorSpec(
                    shape=(None, None, self.sequence_length, self.num_sensors),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                tf.TensorSpec(
                    shape=(None, None, self.sequence_length, self.num_sensors),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            ],
        )
        self.learner._compiled_eval_batch_step = self._compiled_eval_batch_step

    def build_compiled_prototype_memory_batch_step(self) -> None:
        """Build the compiled learned-prototype phase-2 update function.

        This function updates only prototype-memory phase variables.
        """
        self._compiled_prototype_memory_batch_step = tf.function(
            self._train_prototype_memory_batch_step_compiled_impl,
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
        self.learner._compiled_prototype_memory_batch_step = (
            self._compiled_prototype_memory_batch_step
        )

    def compute_model_aux_loss(self, dtype: tf.dtypes.DType) -> tf.Tensor:
        """Return regularization losses added by submodules.

        Args:
            dtype: Output dtype for the summed auxiliary loss.
        """
        if not self.model.losses:
            return tf.constant(0.0, dtype=dtype)
        return tf.add_n([tf.cast(loss, dtype) for loss in self.model.losses])

    def apply_gradients(
        self,
        loss: tf.Tensor,
        tape: tf.GradientTape,
        variables: list[tf.Variable] | None = None,
    ) -> tf.Tensor:
        """Apply gradients for the current model update.

        Args:
            loss: Scalar loss tensor.
            tape: Active gradient tape.
            variables: Optional restricted variable list to update.
        """
        trainable_variables = (
            list(self.model.trainable_variables)
            if variables is None
            else list(variables)
        )
        gradients = tape.gradient(loss, trainable_variables)
        grads_and_vars = [
            (grad, variable)
            for grad, variable in zip(gradients, trainable_variables)
            if grad is not None
        ]
        if not grads_and_vars:
            raise RuntimeError("No gradients were produced for model variables.")

        grads, variables = zip(*grads_and_vars)
        self.optimizer.apply_gradients(zip(grads, variables))
        return loss

    def prototype_phase_trainable_variables(self) -> list[tf.Variable]:
        """Return variables updated during prototype-memory fine-tuning.

        Phase 2 updates only the learned prototype bank. Encoder, CAN,
        logit-scale, and other model variables are intentionally excluded.
        """
        prototype_memory = getattr(self.model, "prototype_memory", None)
        variables: list[tf.Variable] = (
            list(prototype_memory.trainable_variables)
            if prototype_memory is not None
            else []
        )
        seen = set()
        unique_variables = []
        for variable in variables:
            identifier = id(variable)
            if identifier in seen:
                continue
            seen.add(identifier)
            unique_variables.append(variable)
        return unique_variables

    def compute_task_batch_objective(
        self,
        episode_outputs: dict[str, tf.Tensor],
        support_y_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        """Compute per-task objective tensors for batched outputs.

        Args:
            episode_outputs: Task-major model output dictionary.
            support_y_batch: Batched support labels.
            query_y_batch: Batched query labels.
        """
        logits = episode_outputs["logits"]
        per_query_loss = keras.losses.sparse_categorical_crossentropy(
            query_y_batch,
            logits,
            from_logits=True,
        )
        task_losses = tf.reduce_mean(per_query_loss, axis=1)
        model_aux_loss = self.compute_model_aux_loss(dtype=task_losses.dtype)
        can_local_losses = tf.zeros_like(task_losses)
        if (
            float(getattr(self.config, "can_local_loss_weight", 0.0)) > 0
            and "can_local_logits" in episode_outputs
        ):
            local_logits = tf.cast(
                episode_outputs["can_local_logits"], task_losses.dtype
            )
            local_time = tf.shape(local_logits)[2]
            local_labels = tf.tile(
                query_y_batch[:, :, tf.newaxis],
                [1, 1, local_time],
            )
            per_local_loss = keras.losses.sparse_categorical_crossentropy(
                local_labels,
                local_logits,
                from_logits=True,
            )
            can_local_losses = tf.cast(
                self.config.can_local_loss_weight,
                task_losses.dtype,
            ) * tf.reduce_mean(per_local_loss, axis=[1, 2])

        can_margin_losses = tf.zeros_like(task_losses)
        if (
            float(getattr(self.config, "can_margin_loss_weight", 0.0)) > 0
            and "similarity_scores" in episode_outputs
        ):
            true_scores, best_other_scores = self.split_batched_can_scores(
                tf.cast(episode_outputs["similarity_scores"], task_losses.dtype),
                query_y_batch,
            )
            can_score_margins = true_scores - best_other_scores
            per_query_margin_loss = tf.maximum(
                tf.cast(self.config.can_margin_target, task_losses.dtype)
                - can_score_margins,
                tf.constant(0.0, dtype=task_losses.dtype),
            )
            can_margin_losses = tf.cast(
                self.config.can_margin_loss_weight,
                task_losses.dtype,
            ) * tf.reduce_mean(per_query_margin_loss, axis=1)

        return {
            "losses": (
                task_losses
                + model_aux_loss
                + can_local_losses
                + can_margin_losses
            ),
            "task_losses": task_losses,
            "can_local_losses": can_local_losses,
            "can_margin_losses": can_margin_losses,
            "model_aux_loss": model_aux_loss,
        }

    def forward_task(
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
        episode_outputs = self.model.forward_episode(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            training=training,
        )
        logits = episode_outputs["logits"]
        objective_inputs = {
            "logits": logits[tf.newaxis, ...],
        }
        if "can_local_logits" in episode_outputs:
            objective_inputs["can_local_logits"] = episode_outputs["can_local_logits"][
                tf.newaxis, ...
            ]
        if "similarity_scores" in episode_outputs:
            objective_inputs["similarity_scores"] = episode_outputs[
                "similarity_scores"
            ][tf.newaxis, ...]
        objective = self.compute_task_batch_objective(
            objective_inputs,
            support_y[tf.newaxis, ...],
            query_y[tf.newaxis, ...],
        )

        outputs = {
            "logits": logits,
            "loss": objective["losses"][0],
            "task_loss": objective["task_losses"][0],
            "can_local_loss": objective["can_local_losses"][0],
            "can_margin_loss": objective["can_margin_losses"][0],
            "model_aux_loss": objective["model_aux_loss"],
        }
        for key in (
            "support_feature_maps",
            "query_feature_maps",
            "prototype_feature_maps",
            "prototype_support_y",
            "can_proto_attention",
            "can_query_attention",
        ):
            if key in episode_outputs:
                outputs[key] = episode_outputs[key]
        if return_similarity_scores:
            outputs["similarity_scores"] = episode_outputs["similarity_scores"]
        return outputs

    def forward_task_batch(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
        training: bool,
        return_similarity_scores: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Run multiple tasks with batched encoding and per-task losses.

        Args:
            support_x_batch: Task-major support windows.
            support_y_batch: Task-major support labels.
            query_x_batch: Task-major query windows.
            query_y_batch: Task-major query labels.
            training: Whether child layers run in training mode.
            return_similarity_scores: Whether to include similarity scores.
        """
        episode_outputs = self.model.forward_episode_batch(
            support_x=support_x_batch,
            support_y=support_y_batch,
            query_x=query_x_batch,
            training=training,
        )
        logits = episode_outputs["logits"]
        objective = self.compute_task_batch_objective(
            episode_outputs,
            support_y_batch,
            query_y_batch,
        )
        outputs = {
            "logits": logits,
            "losses": objective["losses"],
            "task_losses": objective["task_losses"],
            "can_local_losses": objective["can_local_losses"],
            "can_margin_losses": objective["can_margin_losses"],
            "model_aux_loss": objective["model_aux_loss"],
        }
        for key in (
            "support_feature_maps",
            "query_feature_maps",
            "prototype_feature_maps",
            "prototype_support_y",
            "can_proto_attention",
            "can_query_attention",
        ):
            if key in episode_outputs:
                outputs[key] = episode_outputs[key]
        if return_similarity_scores:
            outputs["similarity_scores"] = episode_outputs["similarity_scores"]
        return outputs

    def train_step(self, support_x, support_y, query_x, query_y):
        """Run one optimizer update on a single episodic task.

        Args:
            support_x: Support windows.
            support_y: Support labels.
            query_x: Query windows.
            query_y: Query labels.
        """
        with tf.GradientTape() as tape:
            task_outputs = self.forward_task(
                support_x=support_x,
                support_y=support_y,
                query_x=query_x,
                query_y=query_y,
                training=True,
            )
            logits = task_outputs["logits"]
            loss = task_outputs["loss"]

        loss = self.apply_gradients(loss, tape)

        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(predictions, tf.cast(query_y, tf.int64)), tf.float32)
        )

        return loss, accuracy

    def evaluate_task(self, support_x, support_y, query_x, query_y):
        """Evaluate one task without updating weights.

        Args:
            support_x: Support windows.
            support_y: Support labels.
            query_x: Query windows.
            query_y: Query labels.
        """
        task_outputs = self.forward_task(
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

    def _eval_task_batch_step_compiled_impl(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ):
        """Evaluate a task batch inside a compiled TensorFlow graph.

        Args:
            support_x_batch: Task-major support windows.
            support_y_batch: Task-major support labels.
            query_x_batch: Task-major query windows.
            query_y_batch: Task-major query labels.
        """
        losses = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False
        )
        task_losses = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False
        )
        can_local_losses = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False
        )
        can_margin_losses = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False
        )
        y_true = tf.TensorArray(tf.int32, size=0, dynamic_size=True, infer_shape=False)
        y_pred = tf.TensorArray(tf.int32, size=0, dynamic_size=True, infer_shape=False)
        intra_scores = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False
        )
        inter_scores = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False
        )
        can_true_scores = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False
        )
        can_best_other_scores = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False
        )
        can_score_margins = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False
        )

        total_tasks = tf.shape(support_x_batch)[0]
        chunk_size = tf.minimum(
            tf.constant(max(1, int(self.task_chunk_size)), dtype=tf.int32),
            total_tasks,
        )

        def _condition(task_start, *_):
            """Return whether compiled eval loop has remaining tasks.

            Args:
                task_start: Current task offset.
            """
            return task_start < total_tasks

        def _body(
            task_start,
            chunk_index,
            losses,
            task_losses,
            can_local_losses,
            can_margin_losses,
            y_true,
            y_pred,
            intra_scores,
            inter_scores,
            can_true_scores,
            can_best_other_scores,
            can_score_margins,
        ):
            """Evaluate one compiled task chunk and append metric tensors.

            Args:
                task_start: Current task offset.
                chunk_index: TensorArray write index for this chunk.
            """
            task_end = tf.minimum(task_start + chunk_size, total_tasks)
            task_outputs = self.forward_task_batch(
                support_x_batch=support_x_batch[task_start:task_end],
                support_y_batch=support_y_batch[task_start:task_end],
                query_x_batch=query_x_batch[task_start:task_end],
                query_y_batch=query_y_batch[task_start:task_end],
                training=False,
                return_similarity_scores=True,
            )
            (
                chunk_losses,
                chunk_task_losses,
                chunk_can_local_losses,
                chunk_can_margin_losses,
                chunk_y_true,
                chunk_y_pred,
                chunk_intra_scores,
                chunk_inter_scores,
                chunk_can_true_scores,
                chunk_can_best_other_scores,
                chunk_can_score_margins,
            ) = self.eval_metric_tensors_from_chunk_outputs(
                task_outputs,
                query_y_batch[task_start:task_end],
            )
            return (
                task_end,
                chunk_index + 1,
                losses.write(chunk_index, chunk_losses),
                task_losses.write(chunk_index, chunk_task_losses),
                can_local_losses.write(chunk_index, chunk_can_local_losses),
                can_margin_losses.write(chunk_index, chunk_can_margin_losses),
                y_true.write(chunk_index, chunk_y_true),
                y_pred.write(chunk_index, chunk_y_pred),
                intra_scores.write(chunk_index, chunk_intra_scores),
                inter_scores.write(chunk_index, chunk_inter_scores),
                can_true_scores.write(chunk_index, chunk_can_true_scores),
                can_best_other_scores.write(chunk_index, chunk_can_best_other_scores),
                can_score_margins.write(chunk_index, chunk_can_score_margins),
            )

        (
            _,
            _,
            losses,
            task_losses,
            can_local_losses,
            can_margin_losses,
            y_true,
            y_pred,
            intra_scores,
            inter_scores,
            can_true_scores,
            can_best_other_scores,
            can_score_margins,
        ) = tf.while_loop(
            _condition,
            _body,
            loop_vars=(
                tf.constant(0, dtype=tf.int32),
                tf.constant(0, dtype=tf.int32),
                losses,
                task_losses,
                can_local_losses,
                can_margin_losses,
                y_true,
                y_pred,
                intra_scores,
                inter_scores,
                can_true_scores,
                can_best_other_scores,
                can_score_margins,
            ),
        )

        return (
            losses.concat(),
            task_losses.concat(),
            can_local_losses.concat(),
            can_margin_losses.concat(),
            y_true.concat(),
            y_pred.concat(),
            intra_scores.concat(),
            inter_scores.concat(),
            can_true_scores.concat(),
            can_best_other_scores.concat(),
            can_score_margins.concat(),
        )

    def _train_batch_step_compiled_impl(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ):
        """Run a compiled optimizer update over episodic task chunks.

        Args:
            support_x_batch: Task-major support windows.
            support_y_batch: Task-major support labels.
            query_x_batch: Task-major query windows.
            query_y_batch: Task-major query labels.
        """
        with tf.GradientTape() as tape:
            losses = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, infer_shape=False
            )
            task_losses = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, infer_shape=False
            )
            accuracies = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, infer_shape=False
            )
            can_local_losses = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, infer_shape=False
            )
            can_margin_losses = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, infer_shape=False
            )

            total_tasks = tf.shape(support_x_batch)[0]
            chunk_size = tf.minimum(
                tf.constant(max(1, int(self.task_chunk_size)), dtype=tf.int32),
                total_tasks,
            )

            def _condition(task_start, *_):
                """Return whether compiled train loop has remaining tasks.

                Args:
                    task_start: Current task offset.
                """
                return task_start < total_tasks

            def _body(
                task_start,
                chunk_index,
                losses,
                task_losses,
                accuracies,
                can_local_losses,
                can_margin_losses,
            ):
                """Train-forward one task chunk and append metric tensors.

                Args:
                    task_start: Current task offset.
                    chunk_index: TensorArray write index for this chunk.
                """
                task_end = tf.minimum(task_start + chunk_size, total_tasks)
                support_x_chunk = self.augment_training_inputs(
                    support_x_batch[task_start:task_end]
                )
                query_x_chunk = self.augment_training_inputs(
                    query_x_batch[task_start:task_end]
                )
                query_y_chunk = query_y_batch[task_start:task_end]
                task_outputs = self.forward_task_batch(
                    support_x_batch=support_x_chunk,
                    support_y_batch=support_y_batch[task_start:task_end],
                    query_x_batch=query_x_chunk,
                    query_y_batch=query_y_chunk,
                    training=True,
                )
                (
                    chunk_losses,
                    chunk_task_losses,
                    chunk_accuracies,
                    chunk_can_local_losses,
                    chunk_can_margin_losses,
                ) = self.train_metric_tensors_from_chunk_outputs(
                    task_outputs,
                    query_y_chunk,
                )
                return (
                    task_end,
                    chunk_index + 1,
                    losses.write(chunk_index, chunk_losses),
                    task_losses.write(chunk_index, chunk_task_losses),
                    accuracies.write(chunk_index, chunk_accuracies),
                    can_local_losses.write(chunk_index, chunk_can_local_losses),
                    can_margin_losses.write(chunk_index, chunk_can_margin_losses),
                )

            (
                _,
                _,
                losses,
                task_losses,
                accuracies,
                can_local_losses,
                can_margin_losses,
            ) = tf.while_loop(
                _condition,
                _body,
                loop_vars=(
                    tf.constant(0, dtype=tf.int32),
                    tf.constant(0, dtype=tf.int32),
                    losses,
                    task_losses,
                    accuracies,
                    can_local_losses,
                    can_margin_losses,
                ),
            )

            batch_loss = tf.reduce_mean(losses.concat())
            batch_task_loss = tf.reduce_mean(task_losses.concat())
            batch_acc = tf.reduce_mean(accuracies.concat())
            batch_can_local_loss = tf.reduce_mean(can_local_losses.concat())
            batch_can_margin_loss = tf.reduce_mean(can_margin_losses.concat())

        batch_loss = self.apply_gradients(batch_loss, tape)
        return (
            batch_loss,
            batch_task_loss,
            batch_acc,
            batch_can_local_loss,
            batch_can_margin_loss,
        )

    def _train_prototype_memory_batch_step_compiled_impl(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ):
        """Run a compiled phase-2 learned-prototype update.

        Args:
            support_x_batch: Task-major support windows.
            support_y_batch: Task-major support labels.
            query_x_batch: Task-major query windows.
            query_y_batch: Task-major query labels.
        """
        if not getattr(self.model, "can_enabled", False):
            raise ValueError("Prototype-memory fine-tuning requires CAN to be enabled")
        original_support_mode = self.model.can_support_mode
        original_encoder_trainable = self.model.encoder.trainable
        original_can_trainable = (
            self.model.cross_attention.trainable
            if getattr(self.model, "cross_attention", None) is not None
            else None
        )
        self.model.can_support_mode = "learned_prototype_memory"
        self.model.encoder.trainable = False
        if getattr(self.model, "cross_attention", None) is not None:
            self.model.cross_attention.trainable = False
        try:
            with tf.GradientTape() as tape:
                task_outputs = self.forward_task_batch(
                    support_x_batch=support_x_batch,
                    support_y_batch=support_y_batch,
                    query_x_batch=query_x_batch,
                    query_y_batch=query_y_batch,
                    training=False,
                )
                batch_loss = tf.reduce_mean(task_outputs["losses"])
                batch_task_loss = tf.reduce_mean(task_outputs["task_losses"])
                batch_can_local_loss = tf.reduce_mean(task_outputs["can_local_losses"])
                batch_can_margin_loss = tf.reduce_mean(
                    task_outputs["can_margin_losses"]
                )
                predictions = tf.argmax(
                    task_outputs["logits"],
                    axis=2,
                    output_type=tf.int32,
                )
                batch_acc = tf.reduce_mean(
                    tf.cast(tf.equal(predictions, query_y_batch), tf.float32)
                )
            batch_loss = self.apply_gradients(
                batch_loss,
                tape,
                variables=self.prototype_phase_trainable_variables(),
            )
            return (
                batch_loss,
                batch_task_loss,
                batch_acc,
                batch_can_local_loss,
                batch_can_margin_loss,
            )
        finally:
            self.model.encoder.trainable = original_encoder_trainable
            if (
                getattr(self.model, "cross_attention", None) is not None
                and original_can_trainable is not None
            ):
                self.model.cross_attention.trainable = original_can_trainable
            self.model.can_support_mode = original_support_mode

    @staticmethod
    def mean_concat(tensor_parts: list[tf.Tensor]) -> tf.Tensor:
        """Return the mean over rank-1 tensors from task chunks.

        Args:
            tensor_parts: List of tensors to concatenate before reducing.
        """
        return tf.reduce_mean(tf.concat(tensor_parts, axis=0))

    def augment_training_task_chunk(
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
                self.augment_training_inputs(support_x_chunk[0])[tf.newaxis, ...],
                self.augment_training_inputs(query_x_chunk[0])[tf.newaxis, ...],
            )
        return (
            self.augment_training_inputs(support_x_chunk),
            self.augment_training_inputs(query_x_chunk),
        )

    def forward_task_chunk(
        self,
        support_x_chunk: tf.Tensor,
        support_y_chunk: tf.Tensor,
        query_x_chunk: tf.Tensor,
        query_y_chunk: tf.Tensor,
        *,
        training: bool,
        return_similarity_scores: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Forward one eager chunk and normalize outputs to task-major tensors.

        Args:
            support_x_chunk: Task-major support windows.
            support_y_chunk: Task-major support labels.
            query_x_chunk: Task-major query windows.
            query_y_chunk: Task-major query labels.
            training: Whether child layers run in training mode.
            return_similarity_scores: Whether to include similarity scores.
        """
        task_count = int(tf.shape(support_x_chunk)[0].numpy())
        if task_count == 1:
            task_outputs = self.forward_task(
                support_x=support_x_chunk[0],
                support_y=support_y_chunk[0],
                query_x=query_x_chunk[0],
                query_y=query_y_chunk[0],
                training=training,
                return_similarity_scores=return_similarity_scores,
            )
            outputs = {
                "logits": task_outputs["logits"][tf.newaxis, ...],
                "losses": tf.reshape(task_outputs["loss"], [1]),
                "task_losses": tf.reshape(task_outputs["task_loss"], [1]),
                "can_local_losses": tf.reshape(task_outputs["can_local_loss"], [1]),
                "can_margin_losses": tf.reshape(task_outputs["can_margin_loss"], [1]),
                "model_aux_loss": task_outputs["model_aux_loss"],
            }
            for key in (
                "support_feature_maps",
                "query_feature_maps",
                "prototype_feature_maps",
                "prototype_support_y",
                "can_proto_attention",
                "can_query_attention",
            ):
                if key in task_outputs:
                    outputs[key] = task_outputs[key][tf.newaxis, ...]
            if return_similarity_scores:
                outputs["similarity_scores"] = task_outputs["similarity_scores"][
                    tf.newaxis, ...
                ]
            return outputs

        return self.forward_task_batch(
            support_x_batch=support_x_chunk,
            support_y_batch=support_y_chunk,
            query_x_batch=query_x_chunk,
            query_y_batch=query_y_chunk,
            training=training,
            return_similarity_scores=return_similarity_scores,
        )

    @staticmethod
    def train_metric_tensors_from_chunk_outputs(
        chunk_outputs: dict[str, tf.Tensor],
        query_y_chunk: tf.Tensor,
    ):
        """Return per-task train losses and accuracies for one chunk.

        Args:
            chunk_outputs: Task-major output dictionary.
            query_y_chunk: Task-major query labels.
        """
        logits = chunk_outputs["logits"]
        predictions = tf.argmax(logits, axis=2, output_type=tf.int32)
        accuracies = tf.reduce_mean(
            tf.cast(tf.equal(predictions, query_y_chunk), tf.float32),
            axis=1,
        )
        return (
            tf.reshape(tf.cast(chunk_outputs["losses"], tf.float32), [-1]),
            tf.reshape(tf.cast(chunk_outputs["task_losses"], tf.float32), [-1]),
            tf.reshape(tf.cast(accuracies, tf.float32), [-1]),
            tf.reshape(tf.cast(chunk_outputs["can_local_losses"], tf.float32), [-1]),
            tf.reshape(tf.cast(chunk_outputs["can_margin_losses"], tf.float32), [-1]),
        )

    def split_batched_similarity_scores(
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

    def eval_metric_tensors_from_chunk_outputs(
        self,
        chunk_outputs: dict[str, tf.Tensor],
        query_y_chunk: tf.Tensor,
    ):
        """Return flattened eval losses, labels, predictions, and scores.

        Args:
            chunk_outputs: Task-major output dictionary.
            query_y_chunk: Task-major query labels.
        """
        logits = chunk_outputs["logits"]
        pred = tf.argmax(logits, axis=2, output_type=tf.int32)
        intra_scores, inter_scores = self.split_batched_similarity_scores(
            chunk_outputs["similarity_scores"],
            query_y_chunk,
        )
        can_true_scores, can_best_other_scores = self.split_batched_can_scores(
            chunk_outputs["similarity_scores"],
            query_y_chunk,
        )
        can_score_margins = can_true_scores - can_best_other_scores
        return (
            tf.reshape(tf.cast(chunk_outputs["losses"], tf.float32), [-1]),
            tf.reshape(tf.cast(chunk_outputs["task_losses"], tf.float32), [-1]),
            tf.reshape(tf.cast(chunk_outputs["can_local_losses"], tf.float32), [-1]),
            tf.reshape(tf.cast(chunk_outputs["can_margin_losses"], tf.float32), [-1]),
            tf.reshape(tf.cast(query_y_chunk, tf.int32), [-1]),
            tf.reshape(pred, [-1]),
            tf.reshape(intra_scores, [-1]),
            tf.reshape(inter_scores, [-1]),
            tf.reshape(can_true_scores, [-1]),
            tf.reshape(can_best_other_scores, [-1]),
            tf.reshape(can_score_margins, [-1]),
        )

    def split_batched_can_scores(
        self,
        similarity_scores: tf.Tensor,
        query_y_batch: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Return true-class and strongest competing CAN scores.

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
        true_scores = tf.gather_nd(
            similarity_scores,
            tf.stack([task_indices, row_indices, query_y_batch], axis=2),
        )
        class_ids = tf.range(int(self.config.n_way), dtype=tf.int32)
        other_mask = tf.not_equal(
            class_ids[tf.newaxis, tf.newaxis, :],
            query_y_batch[:, :, tf.newaxis],
        )
        best_other_scores = tf.reduce_max(
            tf.where(
                other_mask,
                similarity_scores,
                tf.fill(
                    tf.shape(similarity_scores), tf.cast(-1e9, similarity_scores.dtype)
                ),
            ),
            axis=2,
        )
        return true_scores, best_other_scores

    def train_batch_step_eager_tensors(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ):
        """Run an eager optimizer update over a task batch.

        Args:
            support_x_batch: Task-major support windows.
            support_y_batch: Task-major support labels.
            query_x_batch: Task-major query windows.
            query_y_batch: Task-major query labels.
        """
        with tf.GradientTape() as tape:
            losses = []
            task_losses = []
            accuracies = []
            can_local_losses = []
            can_margin_losses = []
            for (
                support_x_chunk,
                support_y_chunk,
                query_x_chunk,
                query_y_chunk,
            ) in self.task_pipeline.iter_task_tensor_chunks(
                support_x_batch,
                support_y_batch,
                query_x_batch,
                query_y_batch,
            ):
                support_x_chunk, query_x_chunk = self.augment_training_task_chunk(
                    support_x_chunk,
                    query_x_chunk,
                )
                chunk_outputs = self.forward_task_chunk(
                    support_x_chunk=support_x_chunk,
                    support_y_chunk=support_y_chunk,
                    query_x_chunk=query_x_chunk,
                    query_y_chunk=query_y_chunk,
                    training=True,
                )
                (
                    chunk_losses,
                    chunk_task_losses,
                    chunk_accuracies,
                    chunk_can_local_losses,
                    chunk_can_margin_losses,
                ) = self.train_metric_tensors_from_chunk_outputs(
                    chunk_outputs,
                    query_y_chunk,
                )
                losses.append(chunk_losses)
                task_losses.append(chunk_task_losses)
                accuracies.append(chunk_accuracies)
                can_local_losses.append(chunk_can_local_losses)
                can_margin_losses.append(chunk_can_margin_losses)

            batch_loss = self.mean_concat(losses)
            batch_task_loss = self.mean_concat(task_losses)
            batch_acc = self.mean_concat(accuracies)
            batch_can_local_loss = self.mean_concat(can_local_losses)
            batch_can_margin_loss = self.mean_concat(can_margin_losses)

        batch_loss = self.apply_gradients(batch_loss, tape)
        return (
            batch_loss,
            batch_task_loss,
            batch_acc,
            batch_can_local_loss,
            batch_can_margin_loss,
        )

    def train_batch_step_tensors(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ):
        """Run the compiled train step with eager fallback.

        Args:
            support_x_batch: Task-major support windows.
            support_y_batch: Task-major support labels.
            query_x_batch: Task-major query windows.
            query_y_batch: Task-major query labels.
        """
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
                self.learner._compiled_train_batch_step = None
        return self.train_batch_step_eager_tensors(
            support_x_batch=support_x_batch,
            support_y_batch=support_y_batch,
            query_x_batch=query_x_batch,
            query_y_batch=query_y_batch,
        )

    def train_prototype_memory_batch_step_tensors(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ):
        """Run compiled prototype-memory update with eager fallback.

        Args:
            support_x_batch: Task-major support windows.
            support_y_batch: Task-major support labels.
            query_x_batch: Task-major query windows.
            query_y_batch: Task-major query labels.
        """
        if self._compiled_prototype_memory_batch_step is not None:
            try:
                return self._compiled_prototype_memory_batch_step(
                    support_x_batch,
                    support_y_batch,
                    query_x_batch,
                    query_y_batch,
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.logger.warning(
                    "Compiled prototype-memory train step failed once; falling back "
                    f"to eager for this batch. error={exc!r}"
                )
                self._compiled_prototype_memory_batch_step = None
                self.learner._compiled_prototype_memory_batch_step = None
        return self.train_prototype_memory_batch_step_eager_tensors(
            support_x_batch=support_x_batch,
            support_y_batch=support_y_batch,
            query_x_batch=query_x_batch,
            query_y_batch=query_y_batch,
        )

    def train_prototype_memory_batch_step_eager_tensors(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ):
        """Run eager phase-2 update using prototype memory support.

        Args:
            support_x_batch: Task-major support windows.
            support_y_batch: Task-major support labels.
            query_x_batch: Task-major query windows.
            query_y_batch: Task-major query labels.
        """
        if not getattr(self.model, "can_enabled", False):
            raise ValueError("Prototype-memory fine-tuning requires CAN to be enabled")
        original_support_mode = self.model.can_support_mode
        original_encoder_trainable = self.model.encoder.trainable
        original_can_trainable = (
            self.model.cross_attention.trainable
            if getattr(self.model, "cross_attention", None) is not None
            else None
        )
        self.model.can_support_mode = "learned_prototype_memory"
        self.model.encoder.trainable = False
        if getattr(self.model, "cross_attention", None) is not None:
            self.model.cross_attention.trainable = False
        try:
            with tf.GradientTape() as tape:
                task_outputs = self.forward_task_batch(
                    support_x_batch=support_x_batch,
                    support_y_batch=support_y_batch,
                    query_x_batch=query_x_batch,
                    query_y_batch=query_y_batch,
                    training=False,
                )
                batch_loss = tf.reduce_mean(task_outputs["losses"])
                batch_task_loss = tf.reduce_mean(task_outputs["task_losses"])
                batch_can_local_loss = tf.reduce_mean(task_outputs["can_local_losses"])
                batch_can_margin_loss = tf.reduce_mean(
                    task_outputs["can_margin_losses"]
                )
                predictions = tf.argmax(
                    task_outputs["logits"],
                    axis=2,
                    output_type=tf.int32,
                )
                batch_acc = tf.reduce_mean(
                    tf.cast(tf.equal(predictions, query_y_batch), tf.float32)
                )
            batch_loss = self.apply_gradients(
                batch_loss,
                tape,
                variables=self.prototype_phase_trainable_variables(),
            )
            return (
                batch_loss,
                batch_task_loss,
                batch_acc,
                batch_can_local_loss,
                batch_can_margin_loss,
            )
        finally:
            self.model.encoder.trainable = original_encoder_trainable
            if (
                getattr(self.model, "cross_attention", None) is not None
                and original_can_trainable is not None
            ):
                self.model.cross_attention.trainable = original_can_trainable
            self.model.can_support_mode = original_support_mode

    def eval_task_batch_step_eager_tensors(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ):
        """Run eager task-batch evaluation without optimizer updates.

        Args:
            support_x_batch: Task-major support windows.
            support_y_batch: Task-major support labels.
            query_x_batch: Task-major query windows.
            query_y_batch: Task-major query labels.
        """
        losses = []
        task_losses = []
        can_local_losses = []
        can_margin_losses = []
        true_labels = []
        pred_labels = []
        intra_scores = []
        inter_scores = []
        can_true_scores = []
        can_best_other_scores = []
        can_score_margins = []

        for (
            support_x_chunk,
            support_y_chunk,
            query_x_chunk,
            query_y_chunk,
        ) in self.task_pipeline.iter_task_tensor_chunks(
            support_x_batch,
            support_y_batch,
            query_x_batch,
            query_y_batch,
        ):
            task_outputs = self.forward_task_chunk(
                support_x_chunk=support_x_chunk,
                support_y_chunk=support_y_chunk,
                query_x_chunk=query_x_chunk,
                query_y_chunk=query_y_chunk,
                training=False,
                return_similarity_scores=True,
            )
            (
                chunk_losses,
                chunk_task_losses,
                chunk_can_local_losses,
                chunk_can_margin_losses,
                chunk_true_labels,
                chunk_pred_labels,
                chunk_intra_scores,
                chunk_inter_scores,
                chunk_can_true_scores,
                chunk_can_best_other_scores,
                chunk_can_score_margins,
            ) = self.eval_metric_tensors_from_chunk_outputs(
                task_outputs,
                query_y_chunk,
            )
            losses.append(chunk_losses)
            task_losses.append(chunk_task_losses)
            can_local_losses.append(chunk_can_local_losses)
            can_margin_losses.append(chunk_can_margin_losses)
            true_labels.append(chunk_true_labels)
            pred_labels.append(chunk_pred_labels)
            intra_scores.append(chunk_intra_scores)
            inter_scores.append(chunk_inter_scores)
            can_true_scores.append(chunk_can_true_scores)
            can_best_other_scores.append(chunk_can_best_other_scores)
            can_score_margins.append(chunk_can_score_margins)

        return (
            tf.concat(losses, axis=0),
            tf.concat(task_losses, axis=0),
            tf.concat(can_local_losses, axis=0),
            tf.concat(can_margin_losses, axis=0),
            tf.concat(true_labels, axis=0),
            tf.concat(pred_labels, axis=0),
            tf.concat(intra_scores, axis=0),
            tf.concat(inter_scores, axis=0),
            tf.concat(can_true_scores, axis=0),
            tf.concat(can_best_other_scores, axis=0),
            tf.concat(can_score_margins, axis=0),
        )

    def eval_task_batch_step_tensors(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ):
        """Run compiled task-batch evaluation with eager fallback.

        Args:
            support_x_batch: Task-major support windows.
            support_y_batch: Task-major support labels.
            query_x_batch: Task-major query windows.
            query_y_batch: Task-major query labels.
        """
        if self._compiled_eval_batch_step is not None:
            try:
                (
                    losses,
                    task_losses,
                    can_local_losses,
                    can_margin_losses,
                    y_true_batch,
                    y_pred_batch,
                    intra_class_scores_batch,
                    inter_class_scores_batch,
                    can_true_scores_batch,
                    can_best_other_scores_batch,
                    can_score_margins_batch,
                ) = self._compiled_eval_batch_step(
                    support_x_batch,
                    support_y_batch,
                    query_x_batch,
                    query_y_batch,
                )
                return (
                    tf.reshape(losses, [-1]),
                    tf.reshape(task_losses, [-1]),
                    tf.reshape(can_local_losses, [-1]),
                    tf.reshape(can_margin_losses, [-1]),
                    tf.reshape(y_true_batch, [-1]),
                    tf.reshape(y_pred_batch, [-1]),
                    tf.reshape(intra_class_scores_batch, [-1]),
                    tf.reshape(inter_class_scores_batch, [-1]),
                    tf.reshape(can_true_scores_batch, [-1]),
                    tf.reshape(can_best_other_scores_batch, [-1]),
                    tf.reshape(can_score_margins_batch, [-1]),
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.logger.warning(
                    "Compiled eval step failed once; falling back to eager for eval batches. "
                    f"error={exc!r}"
                )
                self._compiled_eval_batch_step = None
                self.learner._compiled_eval_batch_step = None
        return self.eval_task_batch_step_eager_tensors(
            support_x_batch=support_x_batch,
            support_y_batch=support_y_batch,
            query_x_batch=query_x_batch,
            query_y_batch=query_y_batch,
        )
