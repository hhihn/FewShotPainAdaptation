import tensorflow as tf
from tensorflow import keras
from utils.logger import setup_logger

from architecture.cnn import EEGNetStyleEncoder


class CrossAttentionModule(keras.layers.Layer):
    """Task-conditioned cross attention over temporal support/query feature maps."""

    def __init__(
        self,
        temperature: float = 1.0,
        meta_hidden_dim: int = 32,
        name: str = "cross_attention_module",
    ):
        super().__init__(name=name)
        self.temperature = float(temperature)
        self.meta_hidden_dim = int(meta_hidden_dim)
        self._built_temporal_shapes = None

    def build(self, input_shape):
        prototype_shape, query_shape = input_shape
        if prototype_shape[-2] is None or query_shape[-2] is None:
            raise ValueError("CAN requires statically known temporal feature-map length")
        if prototype_shape[-1] is None:
            raise ValueError("CAN requires statically known feature-map channel count")
        proto_time = int(prototype_shape[-2])
        query_time = int(query_shape[-2])
        feature_dim = int(prototype_shape[-1])
        descriptor_dim = feature_dim * 4
        self.meta_w1 = self.add_weight(
            name="meta_w1",
            shape=(descriptor_dim, self.meta_hidden_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.meta_b1 = self.add_weight(
            name="meta_b1",
            shape=(self.meta_hidden_dim,),
            initializer="zeros",
            trainable=True,
        )
        self.meta_wp = self.add_weight(
            name="meta_wp",
            shape=(self.meta_hidden_dim, query_time),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.meta_bp = self.add_weight(
            name="meta_bp",
            shape=(query_time,),
            initializer="zeros",
            trainable=True,
        )
        self.meta_wq = self.add_weight(
            name="meta_wq",
            shape=(self.meta_hidden_dim, proto_time),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.meta_bq = self.add_weight(
            name="meta_bq",
            shape=(proto_time,),
            initializer="zeros",
            trainable=True,
        )
        self._built_temporal_shapes = (proto_time, query_time)
        super().build(input_shape)

    @staticmethod
    def _pair_descriptor(
        prototype_maps: tf.Tensor, query_maps: tf.Tensor
    ) -> tf.Tensor:
        proto_summary = tf.reduce_mean(prototype_maps, axis=2)
        query_summary = tf.reduce_mean(query_maps, axis=2)
        proto_pair = proto_summary[:, tf.newaxis, :, :]
        query_pair = query_summary[:, :, tf.newaxis, :]
        proto_pair = tf.broadcast_to(
            proto_pair,
            [
                tf.shape(query_maps)[0],
                tf.shape(query_maps)[1],
                tf.shape(prototype_maps)[1],
                tf.shape(prototype_maps)[3],
            ],
        )
        query_pair = tf.broadcast_to(
            query_pair,
            [
                tf.shape(query_maps)[0],
                tf.shape(query_maps)[1],
                tf.shape(prototype_maps)[1],
                tf.shape(prototype_maps)[3],
            ],
        )
        return tf.concat(
            [
                proto_pair,
                query_pair,
                tf.abs(proto_pair - query_pair),
                proto_pair * query_pair,
            ],
            axis=-1,
        )

    def _meta_kernels(
        self, prototype_maps: tf.Tensor, query_maps: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        descriptor = self._pair_descriptor(prototype_maps, query_maps)
        hidden = tf.nn.gelu(
            tf.einsum("bqcd,dh->bqch", descriptor, self.meta_w1) + self.meta_b1
        )
        kernel_p = tf.nn.softmax(
            tf.einsum("bqch,ht->bqct", hidden, self.meta_wp) + self.meta_bp,
            axis=-1,
        )
        kernel_q = tf.nn.softmax(
            tf.einsum("bqch,ht->bqct", hidden, self.meta_wq) + self.meta_bq,
            axis=-1,
        )
        return kernel_p, kernel_q

    def call(self, inputs) -> dict[str, tf.Tensor]:
        """Return pairwise CAM logits and attention tensors.

        Args:
            inputs: `(prototype_maps, query_maps)`, where prototype maps are
                [tasks, classes, time, channels] and query maps are
                [tasks, queries, time, channels].
        """
        prototype_maps, query_maps = inputs
        temperature = tf.maximum(
            tf.cast(self.temperature, prototype_maps.dtype),
            tf.constant(1e-6, dtype=prototype_maps.dtype),
        )
        proto_norm = tf.nn.l2_normalize(prototype_maps, axis=-1)
        query_norm = tf.nn.l2_normalize(query_maps, axis=-1)
        correlation = tf.einsum("bcpd,bqrd->bqcpr", proto_norm, query_norm)
        kernel_p, kernel_q = self._meta_kernels(prototype_maps, query_maps)
        proto_scores = tf.einsum("bqcpr,bqcr->bqcp", correlation, kernel_p)
        query_scores = tf.einsum("bqcpr,bqcp->bqcr", correlation, kernel_q)
        proto_attention = tf.nn.softmax(proto_scores / temperature, axis=-1)
        query_attention = tf.nn.softmax(query_scores / temperature, axis=-1)

        attended_proto = (
            prototype_maps[:, tf.newaxis, :, :, :]
            * (1.0 + proto_attention[:, :, :, :, tf.newaxis])
        )
        attended_query = (
            query_maps[:, :, tf.newaxis, :, :]
            * (1.0 + query_attention[:, :, :, :, tf.newaxis])
        )
        attended_proto_embeddings = tf.reduce_mean(attended_proto, axis=3)
        attended_query_embeddings = tf.reduce_mean(attended_query, axis=3)
        pairwise_similarity = tf.reduce_sum(
            tf.nn.l2_normalize(attended_proto_embeddings, axis=-1)
            * tf.nn.l2_normalize(attended_query_embeddings, axis=-1),
            axis=-1,
        )
        local_logits = tf.transpose(tf.reduce_max(correlation, axis=3), [0, 1, 3, 2])
        return {
            "similarity_scores": pairwise_similarity,
            "distances": 1.0 - pairwise_similarity,
            "proto_attention": proto_attention,
            "query_attention": query_attention,
            "local_logits": local_logits,
            "correlation": correlation,
        }


class MultimodalPrototypicalNetwork(keras.Model):
    """Prototypical network with a joint EEGNet-style physiological encoder."""

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
        distance_metric: str = "cosine",
        classifier_mode: str = "prototype",
        attention_mode: str = "none",
        can_attention_temperature: float = 1.0,
        can_meta_hidden_dim: int = 32,
        can_transductive_iterations: int = 3,
        can_transductive_top_k_per_class: int = 1,
        can_transductive_min_confidence: float = 0.0,
        seed: int = 0,
    ):
        """
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
        self.can_enabled = self.attention_mode == "can"
        if self.can_enabled and self.classifier_mode != "prototype":
            raise ValueError("attention_mode='can' requires classifier_mode='prototype'")
        if self.can_enabled and self.num_classes < 2:
            raise ValueError("attention_mode='can' requires at least two classes")
        self.eegnet_temporal_filters = int(eegnet_temporal_filters)
        self.eegnet_depth_multiplier = int(eegnet_depth_multiplier)
        self.eegnet_separable_filters = int(eegnet_separable_filters)
        self.eegnet_temporal_kernel_size = int(eegnet_temporal_kernel_size)
        self.eegnet_separable_kernel_size = int(eegnet_separable_kernel_size)
        self.eegnet_pool_size_1 = int(eegnet_pool_size_1)
        self.eegnet_pool_size_2 = int(eegnet_pool_size_2)
        self.eegnet_dropout_rate = float(eegnet_dropout_rate)
        self.eegnet_l2_weight = float(eegnet_l2_weight)
        self.seed = int(seed)
        initial_logit_scale = 10.0 if distance_metric == "cosine" else 1.0
        self.logit_scale = self.add_weight(
            name="logit_scale",
            shape=(),
            initializer=keras.initializers.Constant(initial_logit_scale),
            trainable=True,
            constraint=keras.constraints.NonNeg(),
        )
        self.triplet_centers = self.add_weight(
            name="triplet_centers",
            shape=(self.num_classes, self.embedding_dim),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        self.logger = setup_logger(name="MultimodalPrototypicalNetwork")

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
        )
        self.cross_attention = (
            CrossAttentionModule(
                temperature=self.can_attention_temperature,
                meta_hidden_dim=self.can_meta_hidden_dim,
            )
            if self.can_enabled
            else None
        )
        self.global_classifier = (
            keras.layers.Dense(
                self.num_classes,
                activation=None,
                name="can_global_classifier",
            )
            if self.can_enabled
            else None
        )

        self.logger.debug(
            "Initialized MultimodalPrototypicalNetwork with joint EEGNet encoder"
        )
        self.logger.debug(
            f"Classifier mode: {classifier_mode}, Final embedding dim: {self.fused_embedding_dim}"
        )

    def _log_episode_tensor_stats(self, episode_outputs: dict[str, tf.Tensor]) -> None:
        """Emit lightweight debug stats for one episode."""
        if not self.logger.isEnabledFor(10):
            return
        self.logger.debug(
            "Episode stats: "
            f"support_embeddings_shape={episode_outputs['support_embeddings'].shape}, "
            f"query_embeddings_shape={episode_outputs['query_embeddings'].shape}, "
            f"prototypes_shape={episode_outputs['prototypes'].shape}, "
            f"logits_shape={episode_outputs['logits'].shape}"
        )

    def encode(self, x, training=False):
        """
        Map input to joint EEGNet embedding space.

        Args:
            x: [batch_size, sequence_length, num_sensors]
            training: Whether in training mode

        Returns:
            embeddings: [batch_size, embedding_dim]
        """
        return self.encoder(x, training=training)

    def encode_feature_map(self, x, training=False):
        """Map input windows to temporal EEGNet feature maps."""
        return self.encoder.extract_feature_map(x, training=training)

    def embed_feature_map(self, feature_map, training=False):
        """Pool/project temporal EEGNet feature maps into embedding vectors."""
        return self.encoder.embed_feature_map(feature_map, training=training)

    def _support_normalize_embeddings(self, support_embeddings, query_embeddings):
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
        """Run CAN/CAM over a batch of tasks."""
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
        all_embeddings = self.embed_feature_map(all_feature_maps, training=training)

        support_count = num_tasks * support_size
        support_maps_flat = all_feature_maps[:support_count]
        query_maps_flat = all_feature_maps[support_count:]
        support_embeddings_flat = all_embeddings[:support_count]
        query_embeddings_flat = all_embeddings[support_count:]

        feature_time = tf.shape(support_maps_flat)[1]
        feature_dim = tf.shape(support_maps_flat)[2]
        support_feature_maps = tf.reshape(
            support_maps_flat, [num_tasks, support_size, feature_time, feature_dim]
        )
        query_feature_maps = tf.reshape(
            query_maps_flat, [num_tasks, query_size, feature_time, feature_dim]
        )
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
        prototype_maps = self._compute_prototype_maps_batch(
            support_feature_maps, support_y
        )
        cam_outputs = self.cross_attention((prototype_maps, query_feature_maps))
        similarity_scores = cam_outputs["similarity_scores"]
        distances = cam_outputs["distances"]
        logits = similarity_scores * self.logit_scale
        global_logits = self.global_classifier(query_embeddings, training=training)

        return {
            "support_embeddings": support_embeddings,
            "query_embeddings": query_embeddings,
            "prototypes": prototypes,
            "support_feature_maps": support_feature_maps,
            "query_feature_maps": query_feature_maps,
            "prototype_feature_maps": prototype_maps,
            "distances": distances,
            "logits": logits,
            "similarity_scores": similarity_scores,
            "can_local_logits": cam_outputs["local_logits"],
            "can_global_logits": global_logits,
            "can_proto_attention": cam_outputs["proto_attention"],
            "can_query_attention": cam_outputs["query_attention"],
        }

    def compute_distances(self, query_embeddings, prototype_embeddings):
        """
        Compute distances between query and class prototypes.

        Args:
            query_embeddings: [num_queries, embedding_dim]
            prototype_embeddings: [num_classes, embedding_dim]

        Returns:
            distances: [num_queries, num_classes]
        """
        if self.distance_metric == "euclidean":
          distances = self.compute_cosine_sim(query_embeddings, prototype_embeddings, axis=2)
        elif self.distance_metric == "cosine":
            distances = 1 - self.compute_cosine_sim(query_embeddings, prototype_embeddings)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances

    def compute_distances_batch(self, query_embeddings, prototype_embeddings):
        """
        Compute batched distances between query embeddings and task prototypes.

        Args:
            query_embeddings: [num_tasks, num_queries, embedding_dim]
            prototype_embeddings: [num_tasks, num_classes, embedding_dim]

        Returns:
            distances: [num_tasks, num_queries, num_classes]
        """
        if self.distance_metric == "euclidean":
            distances = self.compute_cosine_sim(query_embeddings, prototype_embeddings, axis=3)
        elif self.distance_metric == "cosine":
            distances = 1 - self.compute_cosine_sim(query_embeddings, prototype_embeddings, axis=2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances

    def compute_similarity_scores(self, query_embeddings, prototype_embeddings):
        """
        Compute similarity scores between query and class prototypes.

        Returns:
            similarities: [num_queries, num_classes]
        """
        if self.distance_metric == "cosine":
            return self.compute_cosine_sim(query_embeddings, prototype_embeddings)

        return -self.compute_distances(query_embeddings, prototype_embeddings)

    def compute_similarity_scores_batch(self, query_embeddings, prototype_embeddings):
        """Compute batched query-to-prototype similarity scores."""
        if self.distance_metric == "cosine":
            return self.compute_cosine_sim(query_embeddings, prototype_embeddings, axis=2)

        return -self.compute_distances_batch(query_embeddings, prototype_embeddings)

    def compute_cosine_sim(self, a_embeddings, b_embeddings, axis=1):
        query_norm = tf.nn.l2_normalize(a_embeddings, axis=axis)
        support_norm = tf.nn.l2_normalize(b_embeddings, axis=axis)
        return tf.matmul(query_norm, support_norm, transpose_b=True)

    def compute_euclidean_sim(self, a_embeddings, b_embeddings, axis=2):
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
        """Compute query-to-support similarities."""
        if self.distance_metric == "cosine":
            return self.compute_cosine_sim(query_embeddings, support_embeddings)
        return -self.compute_pairwise_distances(query_embeddings, support_embeddings)

    def compute_support_to_query_similarity_batch(
        self, query_embeddings: tf.Tensor, support_embeddings: tf.Tensor
    ) -> tf.Tensor:
        """Compute batched query-to-support similarities."""
        if self.distance_metric == "cosine":
            return self.compute_cosine_sim(query_embeddings, support_embeddings, axis=2)

        return -self.compute_pairwise_distances_batch(
            query_embeddings, support_embeddings
        )

    def compute_pairwise_distances(
        self, a_embeddings: tf.Tensor, b_embeddings: tf.Tensor
    ) -> tf.Tensor:
        """Compute pairwise distances between two embedding sets."""
        if self.distance_metric == "euclidean":
            return self.compute_euclidean_sim(a_embeddings, b_embeddings, axis=2)
        if self.distance_metric == "cosine":
            return 1 - self.compute_cosine_sim(a_embeddings, b_embeddings)
        raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def compute_pairwise_distances_batch(
        self, a_embeddings: tf.Tensor, b_embeddings: tf.Tensor
    ) -> tf.Tensor:
        """Compute batched pairwise distances between two embedding sets."""
        if self.distance_metric == "euclidean":
            return self.compute_euclidean_sim(a_embeddings, b_embeddings, axis=3)
        if self.distance_metric == "cosine":
            return 1 - self.compute_cosine_sim(a_embeddings, b_embeddings, axis=2)
        raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _compute_prototypes(self, support_embeddings, support_y):
        """Compute class prototypes as the mean support embedding per class."""
        prototypes = []

        for class_id in range(self.num_classes):
            mask = tf.cast(tf.equal(support_y, class_id), support_embeddings.dtype)
            count = tf.reduce_sum(mask)
            class_embeddings = support_embeddings * tf.expand_dims(mask, 1)
            prototype = tf.reduce_sum(class_embeddings, axis=0) / (count + 1e-8)
            prototypes.append(prototype)

        return tf.stack(prototypes, axis=0)

    def _compute_prototypes_batch(self, support_embeddings, support_y):
        """Compute class prototypes independently for each task in a batch."""
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
        """Aggregate query-to-support similarities into class logits."""
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
        """Aggregate batched query-to-support similarities into class logits."""
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
        """Run one episode and return logits plus intermediate embedding tensors."""
        if self.can_enabled:
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
        class_ids = tf.range(self.num_classes, dtype=support_y.dtype)
        support_class_mask = tf.equal(
            support_y[:, :, tf.newaxis],
            class_ids[tf.newaxis, tf.newaxis, :],
        )
        support_weights = tf.cast(support_class_mask, support_feature_maps.dtype)
        support_sums = tf.einsum("bstd,bsc->bctd", support_feature_maps, support_weights)
        support_counts = tf.reduce_sum(support_weights, axis=1)

        pseudo_class_mask = tf.equal(
            pseudo_labels[:, :, tf.newaxis],
            class_ids[tf.newaxis, tf.newaxis, :],
        )
        pseudo_weights = (
            tf.cast(pseudo_class_mask, query_feature_maps.dtype)
            * tf.cast(selected_mask[:, :, tf.newaxis], query_feature_maps.dtype)
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
        pseudo_labels = tf.argmax(similarity_scores, axis=2, output_type=tf.int32)
        top_values = tf.nn.top_k(similarity_scores, k=2).values
        confidence = top_values[:, :, 0] - top_values[:, :, 1]
        confidence = tf.where(
            selected_mask,
            tf.fill(tf.shape(confidence), tf.constant(-1e9, confidence.dtype)),
            confidence,
        )
        confidence = tf.where(
            confidence >= tf.cast(self.can_transductive_min_confidence, confidence.dtype),
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
        """Run unlabeled query-set transductive CAN inference."""
        if not self.can_enabled or self.can_transductive_iterations <= 0:
            return self.forward_episode_batch(
                support_x=support_x,
                support_y=support_y,
                query_x=query_x,
                training=training,
            )

        outputs = self._forward_episode_batch_can(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            training=training,
        )
        support_feature_maps = outputs["support_feature_maps"]
        query_feature_maps = outputs["query_feature_maps"]
        similarity_scores = outputs["similarity_scores"]
        selected_mask = tf.zeros(tf.shape(similarity_scores)[:2], dtype=tf.bool)
        pseudo_labels = tf.argmax(similarity_scores, axis=2, output_type=tf.int32)

        for _ in range(max(0, int(self.can_transductive_iterations))):
            pseudo_labels, selected_mask = self._select_transductive_pseudo_labels(
                similarity_scores,
                selected_mask,
            )
            prototype_maps = self._prototype_maps_with_pseudo_labels(
                support_feature_maps=support_feature_maps,
                support_y=support_y,
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
        """Run multiple episodes while encoding their samples in one batch."""
        if self.can_enabled:
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
        """
        Forward pass for few-shot learning.

        Args:
            support_x: [n_way * k_shot, sequence_length, num_sensors]
            support_y: [n_way * k_shot] (class labels 0 to n_way - 1)
            query_x: [n_way * q_query, sequence_length, num_sensors]
            training: Whether in training mode

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
