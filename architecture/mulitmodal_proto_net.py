import tensorflow as tf
from tensorflow import keras
from typing import Tuple
from utils.logger import setup_logger

import logging


class MultimodalPrototypicalNetwork(keras.Model):
    """Multimodal Prototypical Networks for few-shot learning on pain data."""

    def __init__(
        self,
        sequence_length: int = 2500,
        num_sensors: int = 3,
        num_classes: int = 6,
        embedding_dim: int = 64,
        modality_names: Tuple[str, ...] = ("EDA", "ECG", "EMG"),
        fusion_method: str = "concat",
        distance_metric: str = "euclidean",
    ):
        """
        Args:
            sequence_length: Length of temporal sequence
            num_sensors: Number of sensor channels
            num_classes: Number of pain levels (6-way)
            embedding_dim: Dimension of embedding space per modality
            modality_names: Names of modalities (EDA, ECG, EMG)
            fusion_method: 'concat', 'mean', 'attention'
            distance_metric: 'euclidean' or 'cosine'
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.num_sensors = num_sensors
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.modality_names = modality_names
        self.fusion_method = fusion_method
        self.distance_metric = distance_metric
        self.logger = setup_logger(
            name="MultimodalPrototypicalNetwork", level=logging.INFO
        )
        # Create separate encoder for each modality
        self.modality_encoders = {}
        for modality_name in modality_names:
            self.modality_encoders[modality_name] = self._build_encoder(
                modality_name, embedding_dim
            )

        # Fusion layer based on fusion method
        if fusion_method == "concat":
            self.fused_embedding_dim = embedding_dim * len(modality_names)
            self.fusion_layer = keras.layers.Dense(embedding_dim, activation="relu")
        elif fusion_method == "mean":
            self.fused_embedding_dim = embedding_dim
            self.fusion_layer = None
        elif fusion_method == "attention":
            self.fused_embedding_dim = embedding_dim * len(modality_names)
            self.attention_weights = keras.layers.Dense(
                len(modality_names), activation="softmax"
            )
            self.fusion_layer = keras.layers.Dense(embedding_dim, activation="relu")
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.logger.info(
            f"Initialized MultimodalPrototypicalNetwork with {len(modality_names)} modalities"
        )
        self.logger.info(
            f"Fusion method: {fusion_method}, Final embedding dim: {self.fused_embedding_dim}"
        )

    def _build_encoder(
        self, modality_name: str, embedding_dim: int
    ) -> keras.Sequential:
        """Build 1D CNN encoder for a single modality."""
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.sequence_length, 1)),
                # Block 1
                keras.layers.Conv1D(
                    32, kernel_size=7, activation="relu", padding="same"
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Conv1D(
                    32, kernel_size=7, activation="relu", padding="same"
                ),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling1D(pool_size=4),
                keras.layers.Dropout(0.3),
                # Block 2
                keras.layers.Conv1D(
                    64, kernel_size=5, activation="relu", padding="same"
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Conv1D(
                    64, kernel_size=5, activation="relu", padding="same"
                ),
                keras.layers.BatchNormalization(),
                keras.layers.MaxPooling1D(pool_size=4),
                keras.layers.Dropout(0.3),
                # Block 3
                keras.layers.Conv1D(
                    128, kernel_size=3, activation="relu", padding="same"
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Conv1D(
                    128, kernel_size=3, activation="relu", padding="same"
                ),
                keras.layers.BatchNormalization(),
                keras.layers.GlobalAveragePooling1D(),
                # Embedding layer
                keras.layers.Dense(embedding_dim, activation="relu"),
                keras.layers.LayerNormalization(),
            ],
            name=f"encoder_{modality_name}",
        )
        self.logger.info(f"Built CNN encoder with {modality_name}")
        self.logger.info(model.summary())

    def encode(self, x, training=False):
        """
        Map input to combined embedding space.

        Args:
            x: [batch_size, sequence_length, num_sensors]
            training: Whether in training mode

        Returns:
            embeddings: [batch_size, fused_embedding_dim] or [batch_size, embedding_dim]
        """
        # Split input by modality
        modality_embeddings = []

        for i, modality_name in enumerate(self.modality_names):
            # Extract single modality: [batch_size, sequence_length, 1]
            modality_data = x[:, :, i : i + 1]

            # Encode modality
            encoder = self.modality_encoders[modality_name]
            embedding = encoder(modality_data, training=training)
            modality_embeddings.append(embedding)

        # Fuse embeddings
        if self.fusion_method == "concat":
            # Concatenate all embeddings
            fused = tf.concat(
                modality_embeddings, axis=1
            )  # [batch, embedding_dim * num_modalities]
            fused = self.fusion_layer(fused)  # [batch, embedding_dim]

        elif self.fusion_method == "mean":
            # Simple mean of embeddings
            fused = tf.stack(
                modality_embeddings, axis=1
            )  # [batch, num_modalities, embedding_dim]
            fused = tf.reduce_mean(fused, axis=1)  # [batch, embedding_dim]

        elif self.fusion_method == "attention":
            # Attention-based fusion
            stacked = tf.stack(
                modality_embeddings, axis=1
            )  # [batch, num_modalities, embedding_dim]

            # Compute attention weights
            # Use mean of modality embeddings as query
            query = tf.reduce_mean(
                stacked, axis=1, keepdims=True
            )  # [batch, 1, embedding_dim]
            scores = tf.reduce_sum(stacked * query, axis=2)  # [batch, num_modalities]
            weights = tf.nn.softmax(scores, axis=1)  # [batch, num_modalities]

            # Apply weights
            weighted = stacked * tf.expand_dims(
                weights, axis=2
            )  # [batch, num_modalities, embedding_dim]
            fused_weighted = tf.reduce_sum(weighted, axis=1)  # [batch, embedding_dim]

            # Concatenate original embeddings with weighted sum for richer representation
            fused = tf.concat(modality_embeddings + [fused_weighted], axis=1)
            fused = self.fusion_layer(fused)  # [batch, embedding_dim]

        return fused

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
            distances = tf.sqrt(
                tf.reduce_sum(
                    (
                        tf.expand_dims(query_embeddings, 1)
                        - tf.expand_dims(prototype_embeddings, 0)
                    )
                    ** 2,
                    axis=2,
                )
                + 1e-8
            )
        elif self.distance_metric == "cosine":
            query_norm = tf.nn.l2_normalize(query_embeddings, axis=1)
            prototype_norm = tf.nn.l2_normalize(prototype_embeddings, axis=1)
            distances = 1 - tf.matmul(query_norm, tf.transpose(prototype_norm))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distances

    def call(self, support_x, support_y, query_x, training=False):
        """
        Forward pass for few-shot learning.

        Args:
            support_x: [n_way * k_shot, sequence_length, num_sensors]
            support_y: [n_way * k_shot] (class labels 0-5)
            query_x: [n_way * q_query, sequence_length, num_sensors]
            training: Whether in training mode

        Returns:
            logits: [n_way * q_query, n_way]
        """
        # Encode all samples
        support_embeddings = self.encode(support_x, training=training)
        query_embeddings = self.encode(query_x, training=training)

        self.logger.debug(f"Support embeddings shape: {tf.shape(support_embeddings)}")
        self.logger.debug(f"Query embeddings shape: {tf.shape(query_embeddings)}")

        # Compute class prototypes as mean of support embeddings per class
        prototypes = []

        for class_id in range(self.num_classes):
            # Create mask for samples belonging to this class
            mask = tf.cast(tf.equal(support_y, class_id), tf.float32)
            count = tf.reduce_sum(mask)

            # Compute mean embedding for this class
            class_embeddings = support_embeddings * tf.expand_dims(mask, 1)
            prototype = tf.reduce_sum(class_embeddings, axis=0) / (count + 1e-8)
            prototypes.append(prototype)

        prototypes = tf.stack(prototypes, axis=0)  # [num_classes, embedding_dim]

        self.logger.debug(f"Prototypes shape: {tf.shape(prototypes)}")

        # Compute distances to prototypes
        distances = self.compute_distances(query_embeddings, prototypes)

        # Convert distances to logits (lower distance = higher probability)
        logits = -distances

        return logits
