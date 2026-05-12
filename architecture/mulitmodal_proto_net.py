import tensorflow as tf
from tensorflow import keras
from utils.logger import setup_logger

from architecture.cnn import EEGNetStyleEncoder


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
        self.logit_scale = 10.0 if distance_metric == "cosine" else 1.0
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

    def compute_cosine_sim(self, a_embeddings, b_embeddings, axis=2):
        query_norm = tf.nn.l2_normalize(a_embeddings, axis=axis)
        support_norm = tf.nn.l2_normalize(b_embeddings, axis=axis)
        return tf.matmul(query_norm, support_norm, transpose_b=True)

    def compute_euclidean_sim(self, a_embeddings, b_embeddings, axis=1):
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
        support_embeddings = self.encode(support_x, training=training)
        query_embeddings = self.encode(query_x, training=training)
        support_mean = tf.reduce_mean(support_embeddings, axis=0, keepdims=True)
        support_std = (
            tf.math.reduce_std(support_embeddings, axis=0, keepdims=True) + 1e-6
        )
        support_embeddings = (support_embeddings - support_mean) / support_std
        query_embeddings = (query_embeddings - support_mean) / support_std

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

    def forward_episode_batch(
        self,
        support_x: tf.Tensor,
        support_y: tf.Tensor,
        query_x: tf.Tensor,
        training: bool = False,
    ) -> dict[str, tf.Tensor]:
        """Run multiple episodes while encoding their samples in one batch."""
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

        support_mean = tf.reduce_mean(support_embeddings, axis=1, keepdims=True)
        support_std = (
            tf.math.reduce_std(support_embeddings, axis=1, keepdims=True) + 1e-6
        )
        support_embeddings = (support_embeddings - support_mean) / support_std
        query_embeddings = (query_embeddings - support_mean) / support_std

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
