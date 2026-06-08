import tensorflow as tf
from tensorflow import keras

from architecture.crossmod_feature_map_encoder import CrossModFeatureMapEncoder
from architecture.eegnet_style_encoder import EEGNetStyleEncoder
from architecture.learned_prototype_memory import LearnedPrototypeMemory
from architecture.crossattention_module import CrossAttentionModule
from utils.logger import setup_logger


class MultimodalPrototypicalNetwork(keras.Model):
    """CAN classifier over temporal physiological feature maps."""

    def __init__(
        self,
        sequence_length: int = 2500,
        num_sensors: int = 3,
        num_classes: int = 6,
        eegnet_temporal_filters: int = 8,
        eegnet_depth_multiplier: int = 2,
        eegnet_separable_filters: int = 16,
        eegnet_temporal_kernel_size: int = 64,
        eegnet_separable_kernel_size: int = 16,
        eegnet_pool_size_1: int = 4,
        eegnet_pool_size_2: int = 8,
        eegnet_dropout_rate: float = 0.25,
        eegnet_l2_weight: float = 1e-4,
        encoder_backend: str = "eegnet",
        crossmod_num_heads: int = 8,
        crossmod_hidden_dim: int = 128,
        crossmod_num_layers: int = 2,
        crossmod_positional_base: float = 10000.0,
        crossmod_attention_dropout_rate: float = 0.0,
        crossmod_ff_activation: str = "relu",
        attention_mode: str = "can",
        can_attention_temperature: float = 1.0,
        can_meta_hidden_dim: int = 32,
        can_support_mode: str = "sampled",
        learned_prototype_slots_per_class: int = 1,
        seed: int = 0,
    ):
        """Initialize a feature-map CAN model.

        Args:
            sequence_length: Length of temporal sequence.
            num_sensors: Number of sensor channels.
            num_classes: Number of task classes.
            attention_mode: Must be ``"can"``.
            can_support_mode: ``"sampled"`` or ``"learned_prototype_memory"``.
        """
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.num_sensors = int(num_sensors)
        self.num_classes = int(num_classes)
        self.attention_mode = str(attention_mode).strip().lower()
        if self.attention_mode != "can":
            raise ValueError("MultimodalPrototypicalNetwork requires attention_mode='can'")
        if self.num_classes < 2:
            raise ValueError("attention_mode='can' requires at least two classes")
        self.can_enabled = True
        self.can_attention_temperature = float(can_attention_temperature)
        self.can_meta_hidden_dim = int(can_meta_hidden_dim)
        self.can_support_mode = str(can_support_mode).strip().lower()
        if self.can_support_mode not in {"sampled", "learned_prototype_memory"}:
            raise ValueError(
                "can_support_mode must be one of: sampled, learned_prototype_memory"
            )
        self.learned_prototype_slots_per_class = int(learned_prototype_slots_per_class)
        if self.learned_prototype_slots_per_class <= 0:
            raise ValueError("learned_prototype_slots_per_class must be > 0")

        self.eegnet_temporal_filters = int(eegnet_temporal_filters)
        self.eegnet_depth_multiplier = int(eegnet_depth_multiplier)
        self.eegnet_separable_filters = int(eegnet_separable_filters)
        self.eegnet_temporal_kernel_size = int(eegnet_temporal_kernel_size)
        self.eegnet_separable_kernel_size = int(eegnet_separable_kernel_size)
        self.eegnet_pool_size_1 = int(eegnet_pool_size_1)
        self.eegnet_pool_size_2 = int(eegnet_pool_size_2)
        self.eegnet_dropout_rate = float(eegnet_dropout_rate)
        self.eegnet_l2_weight = float(eegnet_l2_weight)
        self.encoder_backend = str(encoder_backend).strip().lower()
        if self.encoder_backend not in {"eegnet", "crossmod"}:
            raise ValueError("encoder_backend must be one of: eegnet, crossmod")
        if self.encoder_backend == "crossmod" and self.num_sensors != 2:
            raise ValueError("encoder_backend='crossmod' requires num_sensors=2")

        self.crossmod_num_heads = int(crossmod_num_heads)
        self.crossmod_hidden_dim = int(crossmod_hidden_dim)
        self.crossmod_num_layers = int(crossmod_num_layers)
        self.crossmod_positional_base = float(crossmod_positional_base)
        self.crossmod_attention_dropout_rate = float(crossmod_attention_dropout_rate)
        self.crossmod_ff_activation = str(crossmod_ff_activation)
        self.seed = int(seed)
        self.logger = setup_logger(name="MultimodalPrototypicalNetwork")

        self.logit_scale = self.add_weight(
            name="logit_scale",
            shape=(),
            initializer=keras.initializers.Constant(10.0),
            trainable=True,
            constraint=keras.constraints.NonNeg(),
        )

        if self.encoder_backend == "crossmod":
            self.encoder = CrossModFeatureMapEncoder(
                name="crossmod_encoder",
                sequence_length=self.sequence_length,
                num_sensors=self.num_sensors,
                temporal_filters=self.eegnet_temporal_filters,
                depth_multiplier=self.eegnet_depth_multiplier,
                separable_filters=self.eegnet_separable_filters,
                temporal_kernel_size=self.eegnet_temporal_kernel_size,
                separable_kernel_size=self.eegnet_separable_kernel_size,
                pool_size_1=self.eegnet_pool_size_1,
                pool_size_2=self.eegnet_pool_size_2,
                dropout_rate=self.eegnet_dropout_rate,
                l2_weight=self.eegnet_l2_weight,
                num_heads=self.crossmod_num_heads,
                hidden_dim=self.crossmod_hidden_dim,
                num_layers=self.crossmod_num_layers,
                positional_base=self.crossmod_positional_base,
                attention_dropout_rate=self.crossmod_attention_dropout_rate,
                ff_activation=self.crossmod_ff_activation,
            )
        else:
            self.encoder = EEGNetStyleEncoder(
                name="eegnet_encoder",
                sequence_length=self.sequence_length,
                num_sensors=self.num_sensors,
                temporal_filters=self.eegnet_temporal_filters,
                depth_multiplier=self.eegnet_depth_multiplier,
                separable_filters=self.eegnet_separable_filters,
                temporal_kernel_size=self.eegnet_temporal_kernel_size,
                separable_kernel_size=self.eegnet_separable_kernel_size,
                pool_size_1=self.eegnet_pool_size_1,
                pool_size_2=self.eegnet_pool_size_2,
                dropout_rate=self.eegnet_dropout_rate,
                l2_weight=self.eegnet_l2_weight,
            )
        self.cross_attention = CrossAttentionModule(
            temperature=self.can_attention_temperature,
            meta_hidden_dim=self.can_meta_hidden_dim,
        )
        self.prototype_memory = LearnedPrototypeMemory(
            num_classes=self.num_classes,
            slots_per_class=self.learned_prototype_slots_per_class,
            seed=self.seed + 17,
        )

    def _log_episode_tensor_stats(self, episode_outputs: dict[str, tf.Tensor]) -> None:
        """Emit lightweight tensor-shape diagnostics for one episode."""
        if not self.logger.isEnabledFor(10):
            return
        self.logger.debug(
            "CAN episode stats: "
            f"support_feature_maps_shape={episode_outputs['support_feature_maps'].shape}, "
            f"query_feature_maps_shape={episode_outputs['query_feature_maps'].shape}, "
            f"prototype_feature_maps_shape={episode_outputs['prototype_feature_maps'].shape}, "
            f"logits_shape={episode_outputs['logits'].shape}"
        )

    def encode_feature_map(self, x, training=False):
        """Map input windows to temporal encoder feature maps."""
        return self.encoder.extract_feature_map(x, training=training)

    def _compute_prototype_maps_batch(
        self, support_feature_maps: tf.Tensor, support_y: tf.Tensor
    ) -> tf.Tensor:
        """Compute per-class temporal prototype maps for each task."""
        class_ids = tf.range(self.num_classes, dtype=support_y.dtype)
        class_mask = tf.equal(
            support_y[:, :, tf.newaxis],
            class_ids[tf.newaxis, tf.newaxis, :],
        )
        class_weights = tf.cast(class_mask, support_feature_maps.dtype)
        class_sums = tf.einsum("bstd,bsc->bctd", support_feature_maps, class_weights)
        class_counts = tf.reduce_sum(class_weights, axis=1)
        return class_sums / (class_counts[:, :, tf.newaxis, tf.newaxis] + 1e-8)

    def _forward_episode_batch_can(
        self,
        support_x: tf.Tensor,
        support_y: tf.Tensor,
        query_x: tf.Tensor,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Run CAN/CAM over a batch of sampled-support tasks."""
        num_tasks = tf.shape(support_x)[0]
        support_size = tf.shape(support_x)[1]
        query_size = tf.shape(query_x)[1]
        sequence_length = tf.shape(support_x)[2]
        num_sensors = tf.shape(support_x)[3]

        support_flat = tf.reshape(
            support_x, [num_tasks * support_size, sequence_length, num_sensors]
        )
        query_flat = tf.reshape(
            query_x, [num_tasks * query_size, sequence_length, num_sensors]
        )
        all_feature_maps = self.encode_feature_map(
            tf.concat([support_flat, query_flat], axis=0),
            training=training,
        )

        support_count = num_tasks * support_size
        support_maps_flat = all_feature_maps[:support_count]
        query_maps_flat = all_feature_maps[support_count:]

        feature_time = tf.shape(support_maps_flat)[1]
        feature_dim = tf.shape(support_maps_flat)[2]
        support_feature_maps = tf.reshape(
            support_maps_flat, [num_tasks, support_size, feature_time, feature_dim]
        )
        query_feature_maps = tf.reshape(
            query_maps_flat, [num_tasks, query_size, feature_time, feature_dim]
        )
        prototype_maps = self._compute_prototype_maps_batch(
            support_feature_maps, support_y
        )
        cam_outputs = self.cross_attention((prototype_maps, query_feature_maps))
        similarity_scores = cam_outputs["similarity_scores"]
        logits = similarity_scores * self.logit_scale

        return {
            "support_feature_maps": support_feature_maps,
            "query_feature_maps": query_feature_maps,
            "prototype_feature_maps": prototype_maps,
            "distances": cam_outputs["distances"],
            "logits": logits,
            "similarity_scores": similarity_scores,
            "can_local_logits": cam_outputs["local_logits"],
            "can_proto_attention": cam_outputs["proto_attention"],
            "can_query_attention": cam_outputs["query_attention"],
        }

    def _encode_query_feature_maps(
        self,
        query_x: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Encode batched query windows into temporal feature maps."""
        num_tasks = tf.shape(query_x)[0]
        query_size = tf.shape(query_x)[1]
        sequence_length = tf.shape(query_x)[2]
        num_sensors = tf.shape(query_x)[3]
        query_flat = tf.reshape(
            query_x, [num_tasks * query_size, sequence_length, num_sensors]
        )
        query_feature_maps_flat = self.encode_feature_map(query_flat, training=training)
        feature_time = tf.shape(query_feature_maps_flat)[1]
        feature_dim = tf.shape(query_feature_maps_flat)[2]
        return tf.reshape(
            query_feature_maps_flat,
            [num_tasks, query_size, feature_time, feature_dim],
        )

    def _aggregate_slot_scores(self, slot_scores: tf.Tensor) -> tf.Tensor:
        """Aggregate learned-prototype slot scores into class scores."""
        slot_scores = tf.reshape(
            slot_scores,
            [
                tf.shape(slot_scores)[0],
                tf.shape(slot_scores)[1],
                self.num_classes,
                self.learned_prototype_slots_per_class,
            ],
        )
        return tf.reduce_logsumexp(slot_scores, axis=-1) - tf.math.log(
            tf.cast(self.learned_prototype_slots_per_class, slot_scores.dtype)
        )

    def _aggregate_slot_local_logits(self, local_logits: tf.Tensor) -> tf.Tensor:
        """Aggregate learned-prototype slot local logits into class logits."""
        local_logits = tf.reshape(
            local_logits,
            [
                tf.shape(local_logits)[0],
                tf.shape(local_logits)[1],
                tf.shape(local_logits)[2],
                self.num_classes,
                self.learned_prototype_slots_per_class,
            ],
        )
        return tf.reduce_logsumexp(local_logits, axis=-1) - tf.math.log(
            tf.cast(self.learned_prototype_slots_per_class, local_logits.dtype)
        )

    def _forward_episode_batch_learned_prototype_memory_can(
        self,
        query_x: tf.Tensor,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Run CAN using learned prototype-memory slots as support."""
        query_feature_maps = self._encode_query_feature_maps(query_x, training=training)
        prototype_maps, prototype_y = self.prototype_memory(query_feature_maps)
        cam_outputs = self.cross_attention((prototype_maps, query_feature_maps))
        similarity_scores = self._aggregate_slot_scores(
            cam_outputs["similarity_scores"]
        )
        logits = similarity_scores * self.logit_scale

        return {
            "support_feature_maps": prototype_maps,
            "query_feature_maps": query_feature_maps,
            "prototype_feature_maps": prototype_maps,
            "prototype_support_y": prototype_y,
            "distances": 1.0 - similarity_scores,
            "logits": logits,
            "similarity_scores": similarity_scores,
            "can_local_logits": self._aggregate_slot_local_logits(
                cam_outputs["local_logits"]
            ),
            "can_proto_attention": cam_outputs["proto_attention"],
            "can_query_attention": cam_outputs["query_attention"],
            "slot_similarity_scores": cam_outputs["similarity_scores"],
        }

    def forward_source_subject_prototype_vote_can(
        self,
        *,
        prototype_maps: tf.Tensor,
        prototype_y: tf.Tensor,
        query_x: tf.Tensor,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Run CAN against external source-subject prototypes."""
        prototype_maps = tf.convert_to_tensor(prototype_maps, dtype=tf.float32)
        prototype_y = tf.reshape(
            tf.convert_to_tensor(prototype_y, dtype=tf.int32),
            [-1],
        )
        query_x = tf.convert_to_tensor(query_x, dtype=tf.float32)
        if prototype_maps.shape.rank != 3:
            raise ValueError(
                "prototype_maps must have shape [prototypes, time, channels]"
            )

        query_feature_maps = self._encode_query_feature_maps(
            query_x[tf.newaxis, ...],
            training=training,
        )
        task_prototype_maps = tf.broadcast_to(
            prototype_maps[tf.newaxis, ...],
            [
                1,
                tf.shape(prototype_maps)[0],
                tf.shape(prototype_maps)[1],
                tf.shape(prototype_maps)[2],
            ],
        )
        cam_outputs = self.cross_attention((task_prototype_maps, query_feature_maps))
        prototype_similarity_scores = cam_outputs["similarity_scores"][0]
        prototype_vote_weights = tf.nn.softmax(prototype_similarity_scores, axis=1)
        class_probabilities = tf.math.unsorted_segment_sum(
            data=tf.transpose(prototype_vote_weights, [1, 0]),
            segment_ids=prototype_y,
            num_segments=self.num_classes,
        )
        class_probabilities = tf.transpose(class_probabilities, [1, 0])
        class_probabilities = class_probabilities / (
            tf.reduce_sum(class_probabilities, axis=1, keepdims=True) + 1e-8
        )
        logits = tf.math.log(class_probabilities + 1e-8)

        return {
            "support_feature_maps": task_prototype_maps[0],
            "query_feature_maps": query_feature_maps[0],
            "prototype_feature_maps": task_prototype_maps[0],
            "prototype_support_y": prototype_y,
            "prototype_similarity_scores": prototype_similarity_scores,
            "prototype_vote_weights": prototype_vote_weights,
            "distances": 1.0 - class_probabilities,
            "logits": logits,
            "similarity_scores": class_probabilities,
            "can_local_logits": tf.transpose(
                tf.math.unsorted_segment_sum(
                    data=tf.transpose(cam_outputs["local_logits"][0], [2, 0, 1]),
                    segment_ids=prototype_y,
                    num_segments=self.num_classes,
                ),
                [1, 2, 0],
            ),
            "can_proto_attention": cam_outputs["proto_attention"][0],
            "can_query_attention": cam_outputs["query_attention"][0],
        }

    def forward_episode(self, support_x, support_y, query_x, training=False):
        """Run one few-shot CAN episode."""
        if self.can_support_mode == "learned_prototype_memory":
            batched_outputs = self._forward_episode_batch_learned_prototype_memory_can(
                query_x=query_x[tf.newaxis, ...],
                training=training,
            )
        else:
            batched_outputs = self._forward_episode_batch_can(
                support_x=support_x[tf.newaxis, ...],
                support_y=support_y[tf.newaxis, ...],
                query_x=query_x[tf.newaxis, ...],
                training=training,
            )
        outputs = {
            key: value[0] if isinstance(value, tf.Tensor) else value
            for key, value in batched_outputs.items()
        }
        self._log_episode_tensor_stats(outputs)
        return outputs

    def forward_episode_batch(
        self,
        support_x: tf.Tensor,
        support_y: tf.Tensor,
        query_x: tf.Tensor,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Run multiple CAN episodes while sharing one encoder batch."""
        if self.can_support_mode == "learned_prototype_memory":
            return self._forward_episode_batch_learned_prototype_memory_can(
                query_x=query_x,
                training=training,
            )
        return self._forward_episode_batch_can(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            training=training,
        )

    def call(
        self,
        support_x,
        support_y,
        query_x,
        training=False,
        return_similarity_scores: bool = False,
    ):
        """Run the Keras forward pass for one few-shot episode."""
        episode_outputs = self.forward_episode(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            training=training,
        )
        logits = episode_outputs["logits"]

        if return_similarity_scores:
            return logits, episode_outputs["similarity_scores"]

        return logits
