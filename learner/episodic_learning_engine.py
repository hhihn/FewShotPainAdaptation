import gc

import tensorflow as tf
from tensorflow import keras

from architecture.mulitmodal_proto_net import MultimodalPrototypicalNetwork


class EpisodicLearningEngine:
    """TensorFlow model lifecycle and episodic train/eval tensor execution."""

    def __init__(self, learner):
        self.learner = learner
        self._compiled_train_batch_step = None
        self._compiled_eval_batch_step = None
        self._initial_model_weights = None
        self._initial_optimizer_variables = None

    def __getattr__(self, name):
        return getattr(self.learner, name)

    @property
    def model(self):
        return self.learner.model

    @model.setter
    def model(self, value):
        self.learner.model = value

    @property
    def optimizer(self):
        return self.learner.optimizer

    @optimizer.setter
    def optimizer(self, value):
        self.learner.optimizer = value

    def augment_training_inputs(self, x: tf.Tensor) -> tf.Tensor:
        """Apply training-only signal augmentation configured for episodic updates."""
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
        """Drop TensorFlow model/optimizer references and optionally clear Keras state."""
        self._compiled_train_batch_step = None
        self._compiled_eval_batch_step = None
        self.learner._compiled_train_batch_step = None
        self.learner._compiled_eval_batch_step = None
        self._initial_model_weights = None
        self._initial_optimizer_variables = None
        self.model = None
        self.optimizer = None
        if clear_session:
            tf.keras.backend.clear_session()
        gc.collect()

    def rebuild_model(self, clear_session: bool = True) -> None:
        """Build a fresh model/optimizer, optionally clearing stale TF graph state."""
        if clear_session:
            self.release_model_resources(clear_session=True)

        self.learner.sequence_length = int(self.config.sequence_length)
        self.learner.num_sensors = int(self.config.num_sensors)

        self.model = MultimodalPrototypicalNetwork(
            sequence_length=self.config.sequence_length,
            num_sensors=self.num_sensors,
            num_classes=self.config.n_way,
            embedding_dim=self.embedding_dim,
            distance_metric=self.distance_metric,
            classifier_mode=self.config.classifier_mode,
            attention_mode=self.config.attention_mode,
            can_attention_temperature=self.config.can_attention_temperature,
            can_meta_hidden_dim=self.config.can_meta_hidden_dim,
            can_transductive_iterations=self.config.can_transductive_iterations,
            can_transductive_top_k_per_class=self.config.can_transductive_top_k_per_class,
            can_transductive_min_confidence=self.config.can_transductive_min_confidence,
            eegnet_temporal_filters=self.config.eegnet_temporal_filters,
            eegnet_depth_multiplier=self.config.eegnet_depth_multiplier,
            eegnet_separable_filters=self.config.eegnet_separable_filters,
            eegnet_temporal_kernel_size=self.config.eegnet_temporal_kernel_size,
            eegnet_separable_kernel_size=self.config.eegnet_separable_kernel_size,
            eegnet_pool_size_1=self.config.eegnet_pool_size_1,
            eegnet_pool_size_2=self.config.eegnet_pool_size_2,
            eegnet_dropout_rate=self.config.eegnet_dropout_rate,
            eegnet_l2_weight=self.config.eegnet_l2_weight,
            seed=self.seed,
        )
        optimizer_kwargs = {
            "learning_rate": self.learning_rate,
            "weight_decay": 1e-4,
        }
        if self.gradient_clip_norm is not None:
            optimizer_kwargs["clipnorm"] = self.gradient_clip_norm
        self.optimizer = keras.optimizers.AdamW(**optimizer_kwargs)
        self.initialize_model_and_optimizer_variables()
        self.build_compiled_train_batch_step()
        self.build_compiled_eval_batch_step()
        self.capture_initial_model_state()

    def make_dummy_episode_tensors(
        self,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Create one configured episode to build model and optimizer variables."""
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
        """Build model weights and optimizer slot variables without training."""
        support_x, support_y, query_x, _ = self.make_dummy_episode_tensors()
        self.model.forward_episode(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            training=False,
        )
        if hasattr(self.optimizer, "build"):
            self.optimizer.build(self.model.trainable_variables)

    @staticmethod
    def snapshot_variables(variables: list[tf.Variable]) -> list[tf.Tensor]:
        """Return immutable tensor snapshots for a variable list."""
        return [tf.identity(variable) for variable in variables]

    @staticmethod
    def restore_variable_snapshot(
        variables: list[tf.Variable],
        snapshot: list[tf.Tensor],
        *,
        label: str,
    ) -> None:
        """Assign a previously captured snapshot back into matching variables."""
        if len(variables) != len(snapshot):
            raise RuntimeError(
                f"Cannot restore {label}: variable count changed from "
                f"{len(snapshot)} to {len(variables)}."
            )
        for variable, value in zip(variables, snapshot):
            variable.assign(value)

    def capture_initial_model_state(self) -> None:
        """Capture the post-build untrained model and optimizer state."""
        self._initial_model_weights = self.snapshot_variables(self.model.weights)
        self._initial_optimizer_variables = self.snapshot_variables(
            self.optimizer.variables
        )

    def reset_model_state_for_new_fold(self) -> None:
        """Restore fold-start weights and optimizer state without retracing functions."""
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
        self.learner._compiled_train_batch_step = self._compiled_train_batch_step

    def build_compiled_eval_batch_step(self) -> None:
        """Build a compiled evaluation function for batches of episodic tasks."""
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

    def compute_model_aux_loss(self, dtype: tf.dtypes.DType) -> tf.Tensor:
        """Return regularization losses added by submodules, or zero if absent."""
        if not self.model.losses:
            return tf.constant(0.0, dtype=dtype)
        return tf.add_n([tf.cast(loss, dtype) for loss in self.model.losses])

    def apply_gradients(self, loss: tf.Tensor, tape: tf.GradientTape) -> tf.Tensor:
        """Apply gradients for the current model update."""
        gradients = tape.gradient(loss, self.model.trainable_variables)
        if (
            self.triplet_mining_strategy == "triplet_center"
            and self.triplet_center_gradient_clip_norm > 0
        ):
            center_clip_norm = tf.cast(
                self.triplet_center_gradient_clip_norm,
                tf.float32,
            )
            gradients = [
                (
                    tf.clip_by_norm(grad, center_clip_norm)
                    if grad is not None and variable is self.model.triplet_centers
                    else grad
                )
                for grad, variable in zip(gradients, self.model.trainable_variables)
            ]
        grads_and_vars = [
            (grad, variable)
            for grad, variable in zip(gradients, self.model.trainable_variables)
            if grad is not None
        ]
        if not grads_and_vars:
            raise RuntimeError("No gradients were produced for model variables.")

        grads, variables = zip(*grads_and_vars)
        self.optimizer.apply_gradients(zip(grads, variables))
        return loss

    def compute_batch_all_triplet_loss(
        self, embeddings: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        """Compute BatchAllTripletLoss using cosine distance d(a, b)=1-cos(a, b)."""
        if self.triplet_loss_weight <= 0:
            return tf.constant(0.0, dtype=embeddings.dtype)

        normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        pairwise_similarity = tf.matmul(
            normalized_embeddings, normalized_embeddings, transpose_b=True
        )
        pairwise_distance = tf.ones_like(pairwise_similarity) - pairwise_similarity

        labels = tf.reshape(labels, [-1])
        same_label = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        different_label = tf.logical_not(same_label)
        indices_not_equal = tf.logical_not(tf.eye(tf.shape(labels)[0], dtype=tf.bool))
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
        triplet_loss = tf.maximum(
            triplet_loss,
            tf.constant(0.0, dtype=triplet_loss.dtype),
        )

        positive_triplets = tf.cast(triplet_loss > 1e-16, pairwise_distance.dtype)
        return tf.math.divide_no_nan(
            tf.reduce_sum(triplet_loss),
            tf.reduce_sum(positive_triplets),
        )

    def compute_batch_hard_triplet_loss(
        self, embeddings: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        """Compute BatchHardTripletLoss using cosine distance d(a, b)=1-cos(a, b)."""
        if self.triplet_loss_weight <= 0:
            return tf.constant(0.0, dtype=embeddings.dtype)

        normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        pairwise_similarity = tf.matmul(
            normalized_embeddings, normalized_embeddings, transpose_b=True
        )
        pairwise_distance = tf.ones_like(pairwise_similarity) - pairwise_similarity

        labels = tf.reshape(labels, [-1])
        same_label = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        different_label = tf.logical_not(same_label)
        indices_not_equal = tf.logical_not(tf.eye(tf.shape(labels)[0], dtype=tf.bool))
        positive_mask = tf.logical_and(same_label, indices_not_equal)
        negative_mask = different_label

        valid_anchor_mask = tf.logical_and(
            tf.reduce_any(positive_mask, axis=1),
            tf.reduce_any(negative_mask, axis=1),
        )

        positive_distances = tf.where(
            positive_mask,
            pairwise_distance,
            tf.zeros_like(pairwise_distance),
        )
        hardest_positive_distance = tf.reduce_max(positive_distances, axis=1)

        max_distance = tf.reduce_max(pairwise_distance)
        negative_distances = tf.where(
            negative_mask,
            pairwise_distance,
            tf.ones_like(pairwise_distance)
            * (max_distance + tf.constant(1.0, dtype=pairwise_distance.dtype)),
        )
        hardest_negative_distance = tf.reduce_min(negative_distances, axis=1)

        per_anchor_loss = tf.maximum(
            hardest_positive_distance
            - hardest_negative_distance
            + tf.cast(self.triplet_margin, pairwise_distance.dtype),
            tf.constant(0.0, dtype=pairwise_distance.dtype),
        )
        valid_anchor_weights = tf.cast(valid_anchor_mask, pairwise_distance.dtype)
        return tf.math.divide_no_nan(
            tf.reduce_sum(per_anchor_loss * valid_anchor_weights),
            tf.reduce_sum(valid_anchor_weights),
        )

    def compute_triplet_center_loss(
        self, embeddings: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        """Compute Triplet-Center Loss with trainable per-class centers."""
        if self.triplet_loss_weight <= 0:
            return tf.constant(0.0, dtype=embeddings.dtype)

        centers = tf.cast(self.model.triplet_centers, embeddings.dtype)
        if centers.shape[0] is not None and int(centers.shape[0]) <= 1:
            return tf.constant(0.0, dtype=embeddings.dtype)

        labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
        positive_centers = tf.gather(centers, labels)
        positive_distances = 0.5 * tf.reduce_sum(
            tf.square(embeddings - positive_centers),
            axis=1,
        )

        center_distances = 0.5 * tf.reduce_sum(
            tf.square(embeddings[:, tf.newaxis, :] - centers[tf.newaxis, :, :]),
            axis=2,
        )
        class_ids = tf.range(tf.shape(centers)[0], dtype=labels.dtype)
        negative_mask = tf.not_equal(labels[:, tf.newaxis], class_ids[tf.newaxis, :])
        large_distance = (
            tf.reduce_max(center_distances)
            + tf.constant(1.0, dtype=center_distances.dtype)
        )
        negative_distances = tf.where(
            negative_mask,
            center_distances,
            tf.ones_like(center_distances) * large_distance,
        )
        nearest_negative_distances = tf.reduce_min(negative_distances, axis=1)

        per_sample_loss = tf.maximum(
            positive_distances
            - nearest_negative_distances
            + tf.cast(self.triplet_margin, embeddings.dtype),
            tf.constant(0.0, dtype=embeddings.dtype),
        )
        return tf.reduce_mean(per_sample_loss)

    def compute_triplet_center_loss_batch(
        self,
        embeddings_batch: tf.Tensor,
        labels_batch: tf.Tensor,
    ) -> tf.Tensor:
        """Compute Triplet-Center Loss for a batch of independent episodes."""
        if self.triplet_loss_weight <= 0:
            return tf.zeros(tf.shape(labels_batch)[0], dtype=embeddings_batch.dtype)

        centers = tf.cast(self.model.triplet_centers, embeddings_batch.dtype)
        if centers.shape[0] is not None and int(centers.shape[0]) <= 1:
            return tf.zeros(tf.shape(labels_batch)[0], dtype=embeddings_batch.dtype)

        labels_batch = tf.cast(labels_batch, tf.int32)
        positive_centers = tf.gather(centers, labels_batch)
        positive_distances = 0.5 * tf.reduce_sum(
            tf.square(embeddings_batch - positive_centers),
            axis=2,
        )

        center_distances = 0.5 * tf.reduce_sum(
            tf.square(
                embeddings_batch[:, :, tf.newaxis, :]
                - centers[tf.newaxis, tf.newaxis, :, :]
            ),
            axis=3,
        )
        class_ids = tf.range(tf.shape(centers)[0], dtype=labels_batch.dtype)
        negative_mask = tf.not_equal(
            labels_batch[:, :, tf.newaxis],
            class_ids[tf.newaxis, tf.newaxis, :],
        )
        large_distance = (
            tf.reduce_max(center_distances)
            + tf.constant(1.0, dtype=center_distances.dtype)
        )
        negative_distances = tf.where(
            negative_mask,
            center_distances,
            tf.ones_like(center_distances) * large_distance,
        )
        nearest_negative_distances = tf.reduce_min(negative_distances, axis=2)

        per_sample_loss = tf.maximum(
            positive_distances
            - nearest_negative_distances
            + tf.cast(self.triplet_margin, embeddings_batch.dtype),
            tf.constant(0.0, dtype=embeddings_batch.dtype),
        )
        return tf.reduce_mean(per_sample_loss, axis=1)

    def compute_triplet_loss(
        self, embeddings: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        """Dispatch configured triplet mining strategy over one episode."""
        if self.triplet_mining_strategy == "batch_hard":
            return self.compute_batch_hard_triplet_loss(embeddings, labels)
        if self.triplet_mining_strategy == "batch_all":
            return self.compute_batch_all_triplet_loss(embeddings, labels)
        if self.triplet_mining_strategy == "triplet_center":
            return self.compute_triplet_center_loss(embeddings, labels)
        raise ValueError(
            "triplet_mining_strategy must be one of: "
            "'batch_hard', 'batch_all', 'triplet_center'"
        )

    def compute_task_batch_objective(
        self,
        episode_outputs: dict[str, tf.Tensor],
        support_y_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        """Compute per-task objective tensors for normalized batched episode outputs."""
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
            getattr(self.config, "attention_mode", "none") == "can"
            and float(getattr(self.config, "can_local_loss_weight", 0.0)) > 0
            and "can_local_logits" in episode_outputs
        ):
            local_logits = tf.cast(episode_outputs["can_local_logits"], task_losses.dtype)
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

        can_global_losses = tf.zeros_like(task_losses)
        if (
            getattr(self.config, "attention_mode", "none") == "can"
            and float(getattr(self.config, "can_global_loss_weight", 0.0)) > 0
            and "can_global_logits" in episode_outputs
        ):
            global_logits = tf.cast(
                episode_outputs["can_global_logits"],
                task_losses.dtype,
            )
            per_global_loss = keras.losses.sparse_categorical_crossentropy(
                query_y_batch,
                global_logits,
                from_logits=True,
            )
            can_global_losses = tf.cast(
                self.config.can_global_loss_weight,
                task_losses.dtype,
            ) * tf.reduce_mean(per_global_loss, axis=1)

        embeddings_batch = tf.concat(
            [
                episode_outputs["support_embeddings"],
                episode_outputs["query_embeddings"],
            ],
            axis=1,
        )
        labels_batch = tf.concat([support_y_batch, query_y_batch], axis=1)
        contrastive_losses = tf.zeros_like(task_losses)
        if self.triplet_loss_weight > 0:
            if self.triplet_mining_strategy == "triplet_center":
                triplet_losses = tf.cast(
                    self.triplet_loss_weight,
                    task_losses.dtype,
                ) * self.compute_triplet_center_loss_batch(
                    tf.cast(embeddings_batch, task_losses.dtype),
                    labels_batch,
                )
            else:
                triplet_losses = tf.map_fn(
                    lambda inputs: (
                        tf.cast(self.triplet_loss_weight, task_losses.dtype)
                        * tf.cast(
                            self.compute_triplet_loss(inputs[0], inputs[1]),
                            task_losses.dtype,
                        )
                    ),
                    (embeddings_batch, labels_batch),
                    fn_output_signature=tf.TensorSpec(
                        shape=(), dtype=task_losses.dtype
                    ),
                    parallel_iterations=16,
                )
        else:
            triplet_losses = tf.zeros_like(task_losses)

        return {
            "losses": (
                task_losses
                + model_aux_loss
                + triplet_losses
                + can_local_losses
                + can_global_losses
            ),
            "task_losses": task_losses,
            "contrastive_losses": contrastive_losses,
            "triplet_losses": triplet_losses,
            "can_local_losses": can_local_losses,
            "can_global_losses": can_global_losses,
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
        """Run one task and compute classification plus optional embedding losses."""
        episode_outputs = self.model.forward_episode(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            training=training,
        )
        logits = episode_outputs["logits"]
        objective_inputs = {
            "logits": logits[tf.newaxis, ...],
            "support_embeddings": episode_outputs["support_embeddings"][
                tf.newaxis, ...
            ],
            "query_embeddings": episode_outputs["query_embeddings"][
                tf.newaxis, ...
            ],
        }
        if "can_local_logits" in episode_outputs:
            objective_inputs["can_local_logits"] = episode_outputs["can_local_logits"][
                tf.newaxis, ...
            ]
        if "can_global_logits" in episode_outputs:
            objective_inputs["can_global_logits"] = episode_outputs[
                "can_global_logits"
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
            "contrastive_loss": objective["contrastive_losses"][0],
            "triplet_loss": objective["triplet_losses"][0],
            "can_local_loss": objective["can_local_losses"][0],
            "can_global_loss": objective["can_global_losses"][0],
            "model_aux_loss": objective["model_aux_loss"],
            "support_embeddings": episode_outputs["support_embeddings"],
            "query_embeddings": episode_outputs["query_embeddings"],
            "prototypes": episode_outputs["prototypes"],
        }
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
        """Run multiple tasks with batched embedding and per-task losses."""
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
            "contrastive_losses": objective["contrastive_losses"],
            "triplet_losses": objective["triplet_losses"],
            "can_local_losses": objective["can_local_losses"],
            "can_global_losses": objective["can_global_losses"],
            "model_aux_loss": objective["model_aux_loss"],
            "support_embeddings": episode_outputs["support_embeddings"],
            "query_embeddings": episode_outputs["query_embeddings"],
            "prototypes": episode_outputs["prototypes"],
        }
        if return_similarity_scores:
            outputs["similarity_scores"] = episode_outputs["similarity_scores"]
        return outputs

    def train_step(self, support_x, support_y, query_x, query_y):
        """Single training step on one task."""
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
        """Evaluate on one task without updating weights."""
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
        """Compiled evaluation over vectorized chunks of episodic tasks."""
        losses = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False
        )
        task_losses = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False
        )
        contrastive_losses = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False
        )
        triplet_losses = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False
        )
        can_local_losses = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False
        )
        can_global_losses = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False
        )
        y_true = tf.TensorArray(
            tf.int32, size=0, dynamic_size=True, infer_shape=False
        )
        y_pred = tf.TensorArray(
            tf.int32, size=0, dynamic_size=True, infer_shape=False
        )
        intra_scores = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False
        )
        inter_scores = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False
        )

        total_tasks = tf.shape(support_x_batch)[0]
        chunk_size = tf.minimum(
            tf.constant(max(1, int(self.embedding_batch_size)), dtype=tf.int32),
            total_tasks,
        )

        def _condition(task_start, *_):
            return task_start < total_tasks

        def _body(
            task_start,
            chunk_index,
            losses,
            task_losses,
            contrastive_losses,
            triplet_losses,
            can_local_losses,
            can_global_losses,
            y_true,
            y_pred,
            intra_scores,
            inter_scores,
        ):
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
                chunk_contrastive_losses,
                chunk_triplet_losses,
                chunk_can_local_losses,
                chunk_can_global_losses,
                chunk_y_true,
                chunk_y_pred,
                chunk_intra_scores,
                chunk_inter_scores,
            ) = self.eval_metric_tensors_from_chunk_outputs(
                task_outputs,
                query_y_batch[task_start:task_end],
            )
            return (
                task_end,
                chunk_index + 1,
                losses.write(chunk_index, chunk_losses),
                task_losses.write(chunk_index, chunk_task_losses),
                contrastive_losses.write(chunk_index, chunk_contrastive_losses),
                triplet_losses.write(chunk_index, chunk_triplet_losses),
                can_local_losses.write(chunk_index, chunk_can_local_losses),
                can_global_losses.write(chunk_index, chunk_can_global_losses),
                y_true.write(chunk_index, chunk_y_true),
                y_pred.write(chunk_index, chunk_y_pred),
                intra_scores.write(chunk_index, chunk_intra_scores),
                inter_scores.write(chunk_index, chunk_inter_scores),
            )

        (
            _,
            _,
            losses,
            task_losses,
            contrastive_losses,
            triplet_losses,
            can_local_losses,
            can_global_losses,
            y_true,
            y_pred,
            intra_scores,
            inter_scores,
        ) = tf.while_loop(
            _condition,
            _body,
            loop_vars=(
                tf.constant(0, dtype=tf.int32),
                tf.constant(0, dtype=tf.int32),
                losses,
                task_losses,
                contrastive_losses,
                triplet_losses,
                can_local_losses,
                can_global_losses,
                y_true,
                y_pred,
                intra_scores,
                inter_scores,
            ),
        )

        return (
            losses.concat(),
            task_losses.concat(),
            contrastive_losses.concat(),
            triplet_losses.concat(),
            can_local_losses.concat(),
            can_global_losses.concat(),
            y_true.concat(),
            y_pred.concat(),
            intra_scores.concat(),
            inter_scores.concat(),
        )

    def _train_batch_step_compiled_impl(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ):
        """Compiled optimizer update over vectorized chunks of episodic tasks."""
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
            contrastive_losses = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, infer_shape=False
            )
            triplet_losses = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, infer_shape=False
            )
            can_local_losses = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, infer_shape=False
            )
            can_global_losses = tf.TensorArray(
                tf.float32, size=0, dynamic_size=True, infer_shape=False
            )

            total_tasks = tf.shape(support_x_batch)[0]
            chunk_size = tf.minimum(
                tf.constant(max(1, int(self.embedding_batch_size)), dtype=tf.int32),
                total_tasks,
            )

            def _condition(task_start, *_):
                return task_start < total_tasks

            def _body(
                task_start,
                chunk_index,
                losses,
                task_losses,
                accuracies,
                contrastive_losses,
                triplet_losses,
                can_local_losses,
                can_global_losses,
            ):
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
                    chunk_contrastive_losses,
                    chunk_triplet_losses,
                    chunk_can_local_losses,
                    chunk_can_global_losses,
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
                    contrastive_losses.write(chunk_index, chunk_contrastive_losses),
                    triplet_losses.write(chunk_index, chunk_triplet_losses),
                    can_local_losses.write(chunk_index, chunk_can_local_losses),
                    can_global_losses.write(chunk_index, chunk_can_global_losses),
                )

            (
                _,
                _,
                losses,
                task_losses,
                accuracies,
                contrastive_losses,
                triplet_losses,
                can_local_losses,
                can_global_losses,
            ) = tf.while_loop(
                _condition,
                _body,
                loop_vars=(
                    tf.constant(0, dtype=tf.int32),
                    tf.constant(0, dtype=tf.int32),
                    losses,
                    task_losses,
                    accuracies,
                    contrastive_losses,
                    triplet_losses,
                    can_local_losses,
                    can_global_losses,
                ),
            )

            batch_loss = tf.reduce_mean(losses.concat())
            batch_task_loss = tf.reduce_mean(task_losses.concat())
            batch_acc = tf.reduce_mean(accuracies.concat())
            batch_contrastive_loss = tf.reduce_mean(contrastive_losses.concat())
            batch_triplet_loss = tf.reduce_mean(triplet_losses.concat())
            batch_can_local_loss = tf.reduce_mean(can_local_losses.concat())
            batch_can_global_loss = tf.reduce_mean(can_global_losses.concat())

        batch_loss = self.apply_gradients(batch_loss, tape)
        return (
            batch_loss,
            batch_task_loss,
            batch_acc,
            batch_contrastive_loss,
            batch_triplet_loss,
            batch_can_local_loss,
            batch_can_global_loss,
        )

    @staticmethod
    def mean_concat(tensor_parts: list[tf.Tensor]) -> tf.Tensor:
        """Mean over rank-1 tensors collected from task chunks."""
        return tf.reduce_mean(tf.concat(tensor_parts, axis=0))

    def augment_training_task_chunk(
        self,
        support_x_chunk: tf.Tensor,
        query_x_chunk: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply train augmentation while preserving legacy single-task shapes."""
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
        """Forward one eager task chunk and normalize outputs to task-major tensors."""
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
                "contrastive_losses": tf.reshape(
                    task_outputs["contrastive_loss"], [1]
                ),
                "triplet_losses": tf.reshape(task_outputs["triplet_loss"], [1]),
                "can_local_losses": tf.reshape(task_outputs["can_local_loss"], [1]),
                "can_global_losses": tf.reshape(task_outputs["can_global_loss"], [1]),
                "model_aux_loss": task_outputs["model_aux_loss"],
                "support_embeddings": task_outputs["support_embeddings"][
                    tf.newaxis, ...
                ],
                "query_embeddings": task_outputs["query_embeddings"][tf.newaxis, ...],
                "prototypes": task_outputs["prototypes"][tf.newaxis, ...],
            }
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
        """Return per-task train losses and accuracies for one normalized chunk."""
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
            tf.reshape(tf.cast(chunk_outputs["contrastive_losses"], tf.float32), [-1]),
            tf.reshape(tf.cast(chunk_outputs["triplet_losses"], tf.float32), [-1]),
            tf.reshape(tf.cast(chunk_outputs["can_local_losses"], tf.float32), [-1]),
            tf.reshape(tf.cast(chunk_outputs["can_global_losses"], tf.float32), [-1]),
        )

    def split_batched_similarity_scores(
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

    def eval_metric_tensors_from_chunk_outputs(
        self,
        chunk_outputs: dict[str, tf.Tensor],
        query_y_chunk: tf.Tensor,
    ):
        """Return flattened eval losses, labels, predictions, and similarity groups."""
        logits = chunk_outputs["logits"]
        pred = tf.argmax(logits, axis=2, output_type=tf.int32)
        intra_scores, inter_scores = self.split_batched_similarity_scores(
            chunk_outputs["similarity_scores"],
            query_y_chunk,
        )
        return (
            tf.reshape(tf.cast(chunk_outputs["losses"], tf.float32), [-1]),
            tf.reshape(tf.cast(chunk_outputs["task_losses"], tf.float32), [-1]),
            tf.reshape(tf.cast(chunk_outputs["contrastive_losses"], tf.float32), [-1]),
            tf.reshape(tf.cast(chunk_outputs["triplet_losses"], tf.float32), [-1]),
            tf.reshape(tf.cast(chunk_outputs["can_local_losses"], tf.float32), [-1]),
            tf.reshape(tf.cast(chunk_outputs["can_global_losses"], tf.float32), [-1]),
            tf.reshape(tf.cast(query_y_chunk, tf.int32), [-1]),
            tf.reshape(pred, [-1]),
            tf.reshape(intra_scores, [-1]),
            tf.reshape(inter_scores, [-1]),
        )

    def train_batch_step_eager_tensors(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ):
        """Eager fallback for one optimizer update using a batch of tasks."""
        with tf.GradientTape() as tape:
            losses = []
            task_losses = []
            accuracies = []
            contrastive_losses = []
            triplet_losses = []
            can_local_losses = []
            can_global_losses = []
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
                    chunk_contrastive_losses,
                    chunk_triplet_losses,
                    chunk_can_local_losses,
                    chunk_can_global_losses,
                ) = self.train_metric_tensors_from_chunk_outputs(
                    chunk_outputs,
                    query_y_chunk,
                )
                losses.append(chunk_losses)
                task_losses.append(chunk_task_losses)
                accuracies.append(chunk_accuracies)
                contrastive_losses.append(chunk_contrastive_losses)
                triplet_losses.append(chunk_triplet_losses)
                can_local_losses.append(chunk_can_local_losses)
                can_global_losses.append(chunk_can_global_losses)

            batch_loss = self.mean_concat(losses)
            batch_task_loss = self.mean_concat(task_losses)
            batch_acc = self.mean_concat(accuracies)
            batch_contrastive_loss = self.mean_concat(contrastive_losses)
            batch_triplet_loss = self.mean_concat(triplet_losses)
            batch_can_local_loss = self.mean_concat(can_local_losses)
            batch_can_global_loss = self.mean_concat(can_global_losses)

        batch_loss = self.apply_gradients(batch_loss, tape)
        return (
            batch_loss,
            batch_task_loss,
            batch_acc,
            batch_contrastive_loss,
            batch_triplet_loss,
            batch_can_local_loss,
            batch_can_global_loss,
        )

    def train_batch_step_tensors(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ):
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
                self.learner._compiled_train_batch_step = None
        return self.train_batch_step_eager_tensors(
            support_x_batch=support_x_batch,
            support_y_batch=support_y_batch,
            query_x_batch=query_x_batch,
            query_y_batch=query_y_batch,
        )

    def eval_task_batch_step_eager_tensors(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ):
        """Eager fallback for task-batch evaluation without optimizer updates."""
        losses = []
        task_losses = []
        contrastive_losses = []
        triplet_losses = []
        can_local_losses = []
        can_global_losses = []
        true_labels = []
        pred_labels = []
        intra_scores = []
        inter_scores = []

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
                chunk_contrastive_losses,
                chunk_triplet_losses,
                chunk_can_local_losses,
                chunk_can_global_losses,
                chunk_true_labels,
                chunk_pred_labels,
                chunk_intra_scores,
                chunk_inter_scores,
            ) = self.eval_metric_tensors_from_chunk_outputs(
                task_outputs,
                query_y_chunk,
            )
            losses.append(chunk_losses)
            task_losses.append(chunk_task_losses)
            contrastive_losses.append(chunk_contrastive_losses)
            triplet_losses.append(chunk_triplet_losses)
            can_local_losses.append(chunk_can_local_losses)
            can_global_losses.append(chunk_can_global_losses)
            true_labels.append(chunk_true_labels)
            pred_labels.append(chunk_pred_labels)
            intra_scores.append(chunk_intra_scores)
            inter_scores.append(chunk_inter_scores)

        return (
            tf.concat(losses, axis=0),
            tf.concat(task_losses, axis=0),
            tf.concat(contrastive_losses, axis=0),
            tf.concat(triplet_losses, axis=0),
            tf.concat(can_local_losses, axis=0),
            tf.concat(can_global_losses, axis=0),
            tf.concat(true_labels, axis=0),
            tf.concat(pred_labels, axis=0),
            tf.concat(intra_scores, axis=0),
            tf.concat(inter_scores, axis=0),
        )

    def eval_task_batch_step_tensors(
        self,
        support_x_batch: tf.Tensor,
        support_y_batch: tf.Tensor,
        query_x_batch: tf.Tensor,
        query_y_batch: tf.Tensor,
    ):
        """Run compiled task-batch eval, with eager fallback if compilation fails."""
        if self._compiled_eval_batch_step is not None:
            try:
                (
                    losses,
                    task_losses,
                    contrastive_losses,
                    triplet_losses,
                    can_local_losses,
                    can_global_losses,
                    y_true_batch,
                    y_pred_batch,
                    intra_class_scores_batch,
                    inter_class_scores_batch,
                ) = self._compiled_eval_batch_step(
                    support_x_batch,
                    support_y_batch,
                    query_x_batch,
                    query_y_batch,
                )
                return (
                    tf.reshape(losses, [-1]),
                    tf.reshape(task_losses, [-1]),
                    tf.reshape(contrastive_losses, [-1]),
                    tf.reshape(triplet_losses, [-1]),
                    tf.reshape(can_local_losses, [-1]),
                    tf.reshape(can_global_losses, [-1]),
                    tf.reshape(y_true_batch, [-1]),
                    tf.reshape(y_pred_batch, [-1]),
                    tf.reshape(intra_class_scores_batch, [-1]),
                    tf.reshape(inter_class_scores_batch, [-1]),
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
