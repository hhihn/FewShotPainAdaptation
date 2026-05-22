import tensorflow as tf
from tensorflow import keras
from utils.logger import setup_logger

from architecture.crossmod_feature_map_encoder import CrossModFeatureMapEncoder
from architecture.eegnet_style_encoder import EEGNetStyleEncoder
from architecture.learned_prototype_memory import LearnedPrototypeMemory
from architecture.crossattention_module import CrossAttentionModule


class MultimodalPrototypicalNetwork(keras.Model):
    """Represent episodic physiological samples with a prototypical network.

    The model supports either compact EEGNet embeddings or CrossMod feature maps,
    with optional CAN cross-attention over temporal prototype representations.
    """

    def __init__(
        self,
        sequence_length: int = 2500,
        num_sensors: int = 3,
        num_classes: int = 6,
        embedding_dim: int = 64,
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
        distance_metric: str = "cosine",
        classifier_mode: str = "prototype",
        attention_mode: str = "none",
        can_attention_temperature: float = 1.0,
        can_meta_hidden_dim: int = 32,
        can_transductive_iterations: int = 3,
        can_transductive_top_k_per_class: int = 1,
        can_transductive_min_confidence: float = 0.0,
        can_support_mode: str = "sampled",
        learned_prototype_slots_per_class: int = 1,
        seed: int = 0,
    ):
        """Initialize the multimodal prototypical network.

        Args:
            sequence_length: Length of temporal sequence
            num_sensors: Number of sensor channels
            num_classes: Number of task classes
            embedding_dim: Dimension of joint embedding space
            eegnet_temporal_filters: Number of temporal Conv2D filters
            eegnet_depth_multiplier: Depth multiplier for full-sensor depthwise mixing
            eegnet_separable_filters: Number of separable temporal filters
            eegnet_temporal_kernel_size: Kernel size for the first temporal filters
            eegnet_separable_kernel_size: Kernel size for separable temporal refinement
            eegnet_pool_size_1: Average-pooling factor after depthwise sensor mixing
            eegnet_pool_size_2: Average-pooling factor after separable temporal filtering
            eegnet_dropout_rate: Dropout rate inside the EEGNet encoder
            eegnet_l2_weight: L2 regularization weight on the projection layer
            distance_metric: 'euclidean' or 'cosine'
            classifier_mode: Episodic classifier mode: 'prototype' or 'soft_knn'
        """
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.num_sensors = int(num_sensors)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.fused_embedding_dim = int(embedding_dim)
        self.distance_metric = distance_metric
        self.classifier_mode = classifier_mode
        self.attention_mode = str(attention_mode).strip().lower()
        self.can_attention_temperature = float(can_attention_temperature)
        self.can_meta_hidden_dim = int(can_meta_hidden_dim)
        self.can_transductive_iterations = int(can_transductive_iterations)
        self.can_transductive_top_k_per_class = int(can_transductive_top_k_per_class)
        self.can_transductive_min_confidence = float(can_transductive_min_confidence)
        self.can_support_mode = str(can_support_mode).strip().lower()
        self.learned_prototype_slots_per_class = int(learned_prototype_slots_per_class)
        self.can_enabled = self.attention_mode == "can"
        if self.can_enabled and self.classifier_mode != "prototype":
            raise ValueError(
                "attention_mode='can' requires classifier_mode='prototype'"
            )
        if self.can_enabled and self.num_classes < 2:
            raise ValueError("attention_mode='can' requires at least two classes")
        if self.can_support_mode not in {"sampled", "learned_prototype_memory"}:
            raise ValueError(
                "can_support_mode must be one of: sampled, learned_prototype_memory"
            )
        if self.can_support_mode == "learned_prototype_memory" and not self.can_enabled:
            raise ValueError(
                "can_support_mode='learned_prototype_memory' requires attention_mode='can'"
            )
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
        if self.encoder_backend == "crossmod" and not self.can_enabled:
            raise ValueError("encoder_backend='crossmod' requires attention_mode='can'")
        if self.encoder_backend == "crossmod" and self.num_sensors != 2:
            raise ValueError("encoder_backend='crossmod' requires num_sensors=2")
        self.crossmod_num_heads = int(crossmod_num_heads)
        self.crossmod_hidden_dim = int(crossmod_hidden_dim)
        self.crossmod_num_layers = int(crossmod_num_layers)
        self.crossmod_positional_base = float(crossmod_positional_base)
        self.crossmod_attention_dropout_rate = float(crossmod_attention_dropout_rate)
        self.crossmod_ff_activation = str(crossmod_ff_activation)
        self.seed = int(seed)
        initial_logit_scale = 10.0 if distance_metric == "cosine" else 1.0
        self.logit_scale = self.add_weight(
            name="logit_scale",
            shape=(),
            initializer=keras.initializers.Constant(initial_logit_scale),
            trainable=True,
            constraint=keras.constraints.NonNeg(),
        )
        self.triplet_centers = None
        if not self.can_enabled:
            self.triplet_centers = self.add_weight(
                name="triplet_centers",
                shape=(self.num_classes, self.embedding_dim),
                initializer=keras.initializers.GlorotUniform(seed=self.seed),
                trainable=True,
            )
        self.logger = setup_logger(name="MultimodalPrototypicalNetwork")

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
                embedding_dim=self.embedding_dim,
                temporal_filters=self.eegnet_temporal_filters,
                depth_multiplier=self.eegnet_depth_multiplier,
                separable_filters=self.eegnet_separable_filters,
                temporal_kernel_size=self.eegnet_temporal_kernel_size,
                separable_kernel_size=self.eegnet_separable_kernel_size,
                pool_size_1=self.eegnet_pool_size_1,
                pool_size_2=self.eegnet_pool_size_2,
                dropout_rate=self.eegnet_dropout_rate,
                l2_weight=self.eegnet_l2_weight,
                enable_embedding_projection=not self.can_enabled,
            )
        self.cross_attention = (
            CrossAttentionModule(
                temperature=self.can_attention_temperature,
                meta_hidden_dim=self.can_meta_hidden_dim,
            )
            if self.can_enabled
            else None
        )
        self.prototype_memory = (
            LearnedPrototypeMemory(
                num_classes=self.num_classes,
                slots_per_class=self.learned_prototype_slots_per_class,
                seed=self.seed + 17,
            )
            if self.can_enabled
            else None
        )
        self.global_classifier = None

        self.logger.debug(
            "Initialized MultimodalPrototypicalNetwork with encoder_backend=%s",
            self.encoder_backend,
        )
        self.logger.debug(
            f"Classifier mode: {classifier_mode}, "
            f"embedding_projection_enabled={not self.can_enabled}"
        )

    def _log_episode_tensor_stats(self, episode_outputs: dict[str, tf.Tensor]) -> None:
        """Emit lightweight tensor-shape diagnostics for one episode.

        The method is a no-op unless debug logging is enabled.
        """
        if not self.logger.isEnabledFor(10):
            return
        if self.can_enabled:
            self.logger.debug(
                "CAN episode stats: "
                f"support_feature_maps_shape={episode_outputs['support_feature_maps'].shape}, "
                f"query_feature_maps_shape={episode_outputs['query_feature_maps'].shape}, "
                f"prototype_feature_maps_shape={episode_outputs['prototype_feature_maps'].shape}, "
                f"logits_shape={episode_outputs['logits'].shape}"
            )
        else:
            self.logger.debug(
                "Episode stats: "
                f"support_embeddings_shape={episode_outputs['support_embeddings'].shape}, "
                f"query_embeddings_shape={episode_outputs['query_embeddings'].shape}, "
                f"prototypes_shape={episode_outputs['prototypes'].shape}, "
                f"logits_shape={episode_outputs['logits'].shape}"
            )

    def encode(self, x, training=False):
        """Map input windows to the configured embedding space.

        Args:
            x: [batch_size, sequence_length, num_sensors]
            training: Whether in training mode

        Returns:
            embeddings: [batch_size, embedding_dim]
        """
        return self.encoder(x, training=training)

    def encode_feature_map(self, x, training=False):
        """Map input windows to temporal encoder feature maps.

        This path is used by CAN/CrossMod workflows that operate on feature maps
        rather than pooled embeddings.
        """
        return self.encoder.extract_feature_map(x, training=training)

    def embed_feature_map(self, feature_map, training=False):
        """Pool and project temporal feature maps into embedding vectors.

        The concrete encoder owns the projection behavior and may reject this
        call when embedding projection is disabled.
        """
        return self.encoder.embed_feature_map(feature_map, training=training)

    def _support_normalize_embeddings(self, support_embeddings, query_embeddings):
        """Normalize support and query embeddings from support statistics.

        Statistics are computed over the support-sample axis and then reused for
        the query embeddings to preserve episodic evaluation semantics.
        """
        support_mean = tf.reduce_mean(support_embeddings, axis=-2, keepdims=True)
        support_std = (
            tf.math.reduce_std(support_embeddings, axis=-2, keepdims=True) + 1e-6
        )
        return (
            (support_embeddings - support_mean) / support_std,
            (query_embeddings - support_mean) / support_std,
        )

    def _compute_prototype_maps_batch(
        self, support_feature_maps: tf.Tensor, support_y: tf.Tensor
    ) -> tf.Tensor:
        """Compute per-class temporal prototype maps for each task.

        Each prototype map is the mean of support feature maps assigned to that
        class within the corresponding episodic task.
        """
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
        """Run CAN/CAM over a batch of sampled-support tasks.

        Support and query examples are encoded jointly, class prototype maps are
        computed from support labels, and CrossAttentionModule produces logits.
        """
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
        all_x = tf.concat([support_flat, query_flat], axis=0)
        all_feature_maps = self.encode_feature_map(all_x, training=training)

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
        distances = cam_outputs["distances"]
        logits = similarity_scores * self.logit_scale

        return {
            "support_feature_maps": support_feature_maps,
            "query_feature_maps": query_feature_maps,
            "prototype_feature_maps": prototype_maps,
            "distances": distances,
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
        """Encode batched query windows into temporal feature maps.

        The output preserves the task and query axes while flattening only for
        the encoder call.
        """
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
        """Aggregate learned-prototype slot scores into class scores.

        Scores are reshaped by class and reduced over prototype slots using a
        log-mean-exp aggregation.
        """
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
        """Aggregate learned-prototype slot local logits into class logits.

        Local temporal logits are grouped by class and reduced over prototype
        slots with the same log-mean-exp aggregation used for global scores.
        """
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
        """Run CAN using learned prototype-memory slots as support.

        This path ignores sampled support examples and compares query feature
        maps against trainable class-specific prototype slots.
        """
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

    def compute_distances(self, query_embeddings, prototype_embeddings):
        """Compute distances between query embeddings and class prototypes.

        Args:
            query_embeddings: [num_queries, embedding_dim]
            prototype_embeddings: [num_classes, embedding_dim]

        Returns:
            distances: [num_queries, num_classes]
        """
        if self.distance_metric == "euclidean":
            distances = self.compute_euclidean_sim(
                query_embeddings, prototype_embeddings, axis=2
            )
        elif self.distance_metric == "cosine":
            distances = 1 - self.compute_cosine_sim(
                query_embeddings, prototype_embeddings
            )
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances

    def compute_distances_batch(self, query_embeddings, prototype_embeddings):
        """Compute batched distances between query embeddings and prototypes.

        Args:
            query_embeddings: [num_tasks, num_queries, embedding_dim]
            prototype_embeddings: [num_tasks, num_classes, embedding_dim]

        Returns:
            distances: [num_tasks, num_queries, num_classes]
        """
        if self.distance_metric == "euclidean":
            distances = self.compute_euclidean_sim(
                query_embeddings, prototype_embeddings, axis=3
            )
        elif self.distance_metric == "cosine":
            distances = 1 - self.compute_cosine_sim(
                query_embeddings, prototype_embeddings, axis=2
            )
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances

    def compute_similarity_scores(self, query_embeddings, prototype_embeddings):
        """Compute similarity scores between queries and class prototypes.

        Returns:
            similarities: [num_queries, num_classes]
        """
        if self.distance_metric == "cosine":
            return self.compute_cosine_sim(query_embeddings, prototype_embeddings)

        return -self.compute_distances(query_embeddings, prototype_embeddings)

    def compute_similarity_scores_batch(self, query_embeddings, prototype_embeddings):
        """Compute batched query-to-prototype similarity scores.

        Cosine mode returns normalized dot products; Euclidean mode returns the
        negated configured distance.
        """
        if self.distance_metric == "cosine":
            return self.compute_cosine_sim(
                query_embeddings, prototype_embeddings, axis=2
            )

        return -self.compute_distances_batch(query_embeddings, prototype_embeddings)

    def compute_cosine_sim(self, a_embeddings, b_embeddings, axis=1):
        """Compute cosine similarities between two embedding collections.

        Inputs are L2-normalized along ``axis`` before matrix multiplication.
        """
        query_norm = tf.nn.l2_normalize(a_embeddings, axis=axis)
        support_norm = tf.nn.l2_normalize(b_embeddings, axis=axis)
        return tf.matmul(query_norm, support_norm, transpose_b=True)

    def compute_euclidean_sim(self, a_embeddings, b_embeddings, axis=2):
        """Compute pairwise Euclidean distances between embedding collections.

        The method name is kept for compatibility with existing call sites even
        though the returned value is a distance, not a similarity.
        """
        return tf.sqrt(
            tf.reduce_sum(
                (tf.expand_dims(a_embeddings, 1) - tf.expand_dims(b_embeddings, 0))
                ** 2,
                axis=axis,
            )
            + 1e-8
        )

    def compute_support_to_query_similarity(
        self, query_embeddings: tf.Tensor, support_embeddings: tf.Tensor
    ) -> tf.Tensor:
        """Compute query-to-support similarities for one episode.

        Similarity is cosine similarity or negative pairwise distance depending
        on the configured distance metric.
        """
        if self.distance_metric == "cosine":
            return self.compute_cosine_sim(query_embeddings, support_embeddings)
        return -self.compute_pairwise_distances(query_embeddings, support_embeddings)

    def compute_support_to_query_similarity_batch(
        self, query_embeddings: tf.Tensor, support_embeddings: tf.Tensor
    ) -> tf.Tensor:
        """Compute batched query-to-support similarities.

        The leading task axis is preserved while each query is compared against
        support embeddings from the same task.
        """
        if self.distance_metric == "cosine":
            return self.compute_cosine_sim(query_embeddings, support_embeddings, axis=2)

        return -self.compute_pairwise_distances_batch(
            query_embeddings, support_embeddings
        )

    def compute_pairwise_distances(
        self, a_embeddings: tf.Tensor, b_embeddings: tf.Tensor
    ) -> tf.Tensor:
        """Compute pairwise distances between two embedding sets.

        The selected distance metric controls whether Euclidean distance or
        cosine distance is returned.
        """
        if self.distance_metric == "euclidean":
            return self.compute_euclidean_sim(a_embeddings, b_embeddings, axis=2)
        if self.distance_metric == "cosine":
            return 1 - self.compute_cosine_sim(a_embeddings, b_embeddings)
        raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def compute_pairwise_distances_batch(
        self, a_embeddings: tf.Tensor, b_embeddings: tf.Tensor
    ) -> tf.Tensor:
        """Compute batched pairwise distances between embedding sets.

        Distances are computed independently for each task in the batch.
        """
        if self.distance_metric == "euclidean":
            return self.compute_euclidean_sim(a_embeddings, b_embeddings, axis=3)
        if self.distance_metric == "cosine":
            return 1 - self.compute_cosine_sim(a_embeddings, b_embeddings, axis=2)
        raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _compute_prototypes(self, support_embeddings, support_y):
        """Compute class prototypes for a single episode.

        Each prototype is the mean support embedding for one class.
        """
        prototypes = []

        for class_id in range(self.num_classes):
            mask = tf.cast(tf.equal(support_y, class_id), support_embeddings.dtype)
            count = tf.reduce_sum(mask)
            class_embeddings = support_embeddings * tf.expand_dims(mask, 1)
            prototype = tf.reduce_sum(class_embeddings, axis=0) / (count + 1e-8)
            prototypes.append(prototype)

        return tf.stack(prototypes, axis=0)

    def _compute_prototypes_batch(self, support_embeddings, support_y):
        """Compute class prototypes independently for each batched task.

        The output keeps one prototype tensor per task and per class.
        """
        class_ids = tf.range(self.num_classes, dtype=support_y.dtype)
        class_mask = tf.equal(
            support_y[:, :, tf.newaxis],
            class_ids[tf.newaxis, tf.newaxis, :],
        )
        class_weights = tf.cast(class_mask, support_embeddings.dtype)
        class_sums = tf.einsum("bse,bsc->bce", support_embeddings, class_weights)
        class_counts = tf.reduce_sum(class_weights, axis=1)
        return class_sums / (class_counts[:, :, tf.newaxis] + 1e-8)

    def _compute_soft_knn_logits(
        self,
        support_embeddings: tf.Tensor,
        support_y: tf.Tensor,
        query_embeddings: tf.Tensor,
    ) -> tf.Tensor:
        """Aggregate query-to-support similarities into class logits.

        The single-episode soft k-NN classifier pools support similarities by
        class using log-sum-exp.
        """
        support_similarities = self.compute_support_to_query_similarity(
            query_embeddings=query_embeddings,
            support_embeddings=support_embeddings,
        )
        class_scores = []
        for class_id in range(self.num_classes):
            class_mask = tf.cast(
                tf.equal(support_y, class_id), support_similarities.dtype
            )
            class_scores.append(
                tf.reduce_logsumexp(
                    support_similarities
                    + tf.expand_dims(tf.math.log(class_mask + 1e-8), axis=0),
                    axis=1,
                )
            )
        return tf.stack(class_scores, axis=1)

    def _compute_soft_knn_logits_batch(
        self,
        support_embeddings: tf.Tensor,
        support_y: tf.Tensor,
        query_embeddings: tf.Tensor,
    ) -> tf.Tensor:
        """Aggregate batched query-to-support similarities into logits.

        The task axis is preserved while support similarities are pooled by
        class for each query example.
        """
        support_similarities = self.compute_support_to_query_similarity_batch(
            query_embeddings=query_embeddings,
            support_embeddings=support_embeddings,
        )
        class_ids = tf.range(self.num_classes, dtype=support_y.dtype)
        class_mask = tf.equal(
            support_y[:, :, tf.newaxis],
            class_ids[tf.newaxis, tf.newaxis, :],
        )
        class_mask = tf.cast(class_mask, support_similarities.dtype)
        return tf.reduce_logsumexp(
            support_similarities[:, :, :, tf.newaxis]
            + tf.math.log(class_mask[:, tf.newaxis, :, :] + 1e-8),
            axis=2,
        )

    def forward_episode(self, support_x, support_y, query_x, training=False):
        """Run one few-shot episode.

        The returned dictionary includes logits and intermediate tensors needed
        by losses, diagnostics, and evaluation code.
        """
        if self.can_enabled:
            if self.can_support_mode == "learned_prototype_memory":
                batched_outputs = (
                    self._forward_episode_batch_learned_prototype_memory_can(
                        query_x=query_x[tf.newaxis, ...],
                        training=training,
                    )
                )
            else:
                batched_outputs = self._forward_episode_batch_can(
                    support_x=support_x[tf.newaxis, ...],
                    support_y=support_y[tf.newaxis, ...],
                    query_x=query_x[tf.newaxis, ...],
                    training=training,
                )
            return {
                key: value[0] if isinstance(value, tf.Tensor) else value
                for key, value in batched_outputs.items()
            }

        support_embeddings = self.encode(support_x, training=training)
        query_embeddings = self.encode(query_x, training=training)
        support_embeddings, query_embeddings = self._support_normalize_embeddings(
            support_embeddings, query_embeddings
        )

        if self.logger.isEnabledFor(10):
            self.logger.debug("Support embeddings shape: %s", support_embeddings.shape)
            self.logger.debug("Query embeddings shape: %s", query_embeddings.shape)

        prototypes = self._compute_prototypes(support_embeddings, support_y)
        if self.logger.isEnabledFor(10):
            self.logger.debug("Prototypes shape: %s", prototypes.shape)
        distances = self.compute_distances(query_embeddings, prototypes)

        if self.classifier_mode == "prototype":
            logits = -distances * self.logit_scale
            similarity_scores = self.compute_similarity_scores(
                query_embeddings, prototypes
            )
        elif self.classifier_mode == "soft_knn":
            logits = (
                self._compute_soft_knn_logits(
                    support_embeddings=support_embeddings,
                    support_y=support_y,
                    query_embeddings=query_embeddings,
                )
                * self.logit_scale
            )
            similarity_scores = self.compute_similarity_scores(
                query_embeddings, prototypes
            )
        else:
            raise ValueError(f"Unknown classifier mode: {self.classifier_mode}")

        self._log_episode_tensor_stats(
            {
                "support_embeddings": support_embeddings,
                "query_embeddings": query_embeddings,
                "prototypes": prototypes,
                "logits": logits,
            }
        )

        return {
            "support_embeddings": support_embeddings,
            "query_embeddings": query_embeddings,
            "prototypes": prototypes,
            "distances": distances,
            "logits": logits,
            "similarity_scores": similarity_scores,
        }

    def _prototype_maps_with_pseudo_labels(
        self,
        support_feature_maps: tf.Tensor,
        support_y: tf.Tensor,
        query_feature_maps: tf.Tensor,
        pseudo_labels: tf.Tensor,
        selected_mask: tf.Tensor,
    ) -> tf.Tensor:
        """Recompute prototype maps with selected query pseudo-labels.

        Support labels always contribute, while query maps contribute only where
        the transductive selection mask is true.
        """
        class_ids = tf.range(self.num_classes, dtype=support_y.dtype)
        support_class_mask = tf.equal(
            support_y[:, :, tf.newaxis],
            class_ids[tf.newaxis, tf.newaxis, :],
        )
        support_weights = tf.cast(support_class_mask, support_feature_maps.dtype)
        support_sums = tf.einsum(
            "bstd,bsc->bctd", support_feature_maps, support_weights
        )
        support_counts = tf.reduce_sum(support_weights, axis=1)

        pseudo_class_mask = tf.equal(
            pseudo_labels[:, :, tf.newaxis],
            class_ids[tf.newaxis, tf.newaxis, :],
        )
        pseudo_weights = tf.cast(pseudo_class_mask, query_feature_maps.dtype) * tf.cast(
            selected_mask[:, :, tf.newaxis], query_feature_maps.dtype
        )
        pseudo_sums = tf.einsum("bqtd,bqc->bctd", query_feature_maps, pseudo_weights)
        pseudo_counts = tf.reduce_sum(pseudo_weights, axis=1)
        counts = support_counts + pseudo_counts
        return (support_sums + pseudo_sums) / (
            counts[:, :, tf.newaxis, tf.newaxis] + 1e-8
        )

    def _select_transductive_pseudo_labels(
        self,
        similarity_scores: tf.Tensor,
        selected_mask: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Select confident pseudo-labels for transductive CAN updates.

        The method selects up to the configured top-k queries per predicted class
        while avoiding queries already selected in earlier iterations.
        """
        pseudo_labels = tf.argmax(similarity_scores, axis=2, output_type=tf.int32)
        top_values = tf.nn.top_k(similarity_scores, k=2).values
        confidence = top_values[:, :, 0] - top_values[:, :, 1]
        confidence = tf.where(
            selected_mask,
            tf.fill(tf.shape(confidence), tf.constant(-1e9, confidence.dtype)),
            confidence,
        )
        confidence = tf.where(
            confidence
            >= tf.cast(self.can_transductive_min_confidence, confidence.dtype),
            confidence,
            tf.fill(tf.shape(confidence), tf.constant(-1e9, confidence.dtype)),
        )
        query_size = tf.shape(similarity_scores)[1]
        top_k = tf.minimum(
            query_size,
            tf.constant(
                max(1, int(self.can_transductive_top_k_per_class)),
                dtype=tf.int32,
            ),
        )
        class_selected_masks = []
        for class_id in range(self.num_classes):
            class_scores = tf.where(
                tf.equal(pseudo_labels, class_id),
                confidence,
                tf.fill(tf.shape(confidence), tf.constant(-1e9, confidence.dtype)),
            )
            selected_values, selected_indices = tf.nn.top_k(
                class_scores,
                k=top_k,
            )
            valid = selected_values > tf.constant(-1e8, selected_values.dtype)
            class_selected = tf.reduce_any(
                (tf.one_hot(selected_indices, depth=query_size, dtype=tf.int32) > 0)
                & valid[:, :, tf.newaxis],
                axis=1,
            )
            class_selected_masks.append(class_selected)
        new_selected_mask = selected_mask | tf.reduce_any(
            tf.stack(class_selected_masks, axis=0),
            axis=0,
        )
        return pseudo_labels, new_selected_mask

    def forward_episode_batch_transductive(
        self,
        support_x: tf.Tensor,
        support_y: tf.Tensor,
        query_x: tf.Tensor,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Run transductive CAN inference over unlabeled query sets.

        When transduction is disabled, this method falls back to the standard
        batched episode forward pass.
        """
        if not self.can_enabled or self.can_transductive_iterations <= 0:
            return self.forward_episode_batch(
                support_x=support_x,
                support_y=support_y,
                query_x=query_x,
                training=training,
            )

        if self.can_support_mode == "learned_prototype_memory":
            outputs = self._forward_episode_batch_learned_prototype_memory_can(
                support_x=support_x,
                support_y=support_y,
                query_x=query_x,
                training=training,
            )
        else:
            outputs = self._forward_episode_batch_can(
                support_x=support_x,
                support_y=support_y,
                query_x=query_x,
                training=training,
            )
        support_feature_maps = outputs["support_feature_maps"]
        query_feature_maps = outputs["query_feature_maps"]
        similarity_scores = outputs["similarity_scores"]
        transductive_support_y = outputs.get("prototype_support_y", support_y)
        selected_mask = tf.zeros(tf.shape(similarity_scores)[:2], dtype=tf.bool)
        pseudo_labels = tf.argmax(similarity_scores, axis=2, output_type=tf.int32)

        for _ in range(max(0, int(self.can_transductive_iterations))):
            pseudo_labels, selected_mask = self._select_transductive_pseudo_labels(
                similarity_scores,
                selected_mask,
            )
            prototype_maps = self._prototype_maps_with_pseudo_labels(
                support_feature_maps=support_feature_maps,
                support_y=transductive_support_y,
                query_feature_maps=query_feature_maps,
                pseudo_labels=pseudo_labels,
                selected_mask=selected_mask,
            )
            cam_outputs = self.cross_attention((prototype_maps, query_feature_maps))
            similarity_scores = cam_outputs["similarity_scores"]

        outputs["transductive_similarity_scores"] = similarity_scores
        outputs["transductive_logits"] = similarity_scores * self.logit_scale
        outputs["transductive_selected_mask"] = selected_mask
        outputs["transductive_pseudo_labels"] = pseudo_labels
        return outputs

    def forward_episode_batch(
        self,
        support_x: tf.Tensor,
        support_y: tf.Tensor,
        query_x: tf.Tensor,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Run multiple episodes while sharing one encoder batch.

        The method keeps task structure around a single flattened encoder call
        to reduce repeated TensorFlow overhead.
        """
        if self.can_enabled:
            if self.can_support_mode == "learned_prototype_memory":
                return self._forward_episode_batch_learned_prototype_memory_can(
                    support_x=support_x,
                    support_y=support_y,
                    query_x=query_x,
                    training=training,
                )
            return self._forward_episode_batch_can(
                support_x=support_x,
                support_y=support_y,
                query_x=query_x,
                training=training,
            )

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
        all_x = tf.concat([support_flat, query_flat], axis=0)
        all_embeddings = self.encode(all_x, training=training)
        support_count = num_tasks * support_size
        support_embeddings_flat = all_embeddings[:support_count]
        query_embeddings_flat = all_embeddings[support_count:]
        fused_embedding_dim = tf.shape(support_embeddings_flat)[1]
        support_embeddings = tf.reshape(
            support_embeddings_flat, [num_tasks, support_size, fused_embedding_dim]
        )
        query_embeddings = tf.reshape(
            query_embeddings_flat, [num_tasks, query_size, fused_embedding_dim]
        )

        support_embeddings, query_embeddings = self._support_normalize_embeddings(
            support_embeddings, query_embeddings
        )

        prototypes = self._compute_prototypes_batch(support_embeddings, support_y)
        distances = self.compute_distances_batch(query_embeddings, prototypes)

        if self.classifier_mode == "prototype":
            logits = -distances * self.logit_scale
            similarity_scores = self.compute_similarity_scores_batch(
                query_embeddings, prototypes
            )
        elif self.classifier_mode == "soft_knn":
            logits = (
                self._compute_soft_knn_logits_batch(
                    support_embeddings=support_embeddings,
                    support_y=support_y,
                    query_embeddings=query_embeddings,
                )
                * self.logit_scale
            )
            similarity_scores = self.compute_similarity_scores_batch(
                query_embeddings, prototypes
            )
        else:
            raise ValueError(f"Unknown classifier mode: {self.classifier_mode}")

        return {
            "support_embeddings": support_embeddings,
            "query_embeddings": query_embeddings,
            "prototypes": prototypes,
            "distances": distances,
            "logits": logits,
            "similarity_scores": similarity_scores,
        }

    def call(
        self,
        support_x,
        support_y,
        query_x,
        training=False,
        return_similarity_scores: bool = False,
    ):
        """Run the Keras forward pass for one few-shot episode.

        Args:
            support_x: [n_way * k_shot, sequence_length, num_sensors]
            support_y: [n_way * k_shot] (class labels 0 to n_way - 1)
            query_x: [n_way * q_query, sequence_length, num_sensors]
            training: Whether in training mode
            return_similarity_scores: Whether to return similarities with logits

        Returns:
            logits: [n_way * q_query, n_way]
        """
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
