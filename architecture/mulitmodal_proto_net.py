import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Optional, List
from utils.logger import setup_logger

from architecture.tcn import TemporalConvolutionalNetwork
from architecture.fusion_transformer_ib import TransformerInformationBottleneckFusion


class MultimodalPrototypicalNetwork(keras.Model):
    """Multimodal Prototypical Networks for few-shot learning on pain data."""

    def __init__(
        self,
        sequence_length: int = 2500,
        num_sensors: int = 3,
        num_classes: int = 6,
        embedding_dim: int = 64,
        num_tcn_blocks: int = 3,
        tcn_dilation_rates: Optional[List[int]] = None,
        tcn_kernel_size: int = 3,
        strides: int = 2,
        pooling_size: int = 2,
        filters_list: Optional[List[int]] = None,
        tcn_dropout_rate: float = 0.3,
        tcn_attention_heads: int = 4,
        tcn_attention_key_dim: int = 32,
        tcn_attention_dropout: float = 0.2,
        tcn_attention_pool_size: int = 8,
        use_attention: bool = False,
        modality_names: Tuple[str, ...] = ("EDA", "ECG", "EMG"),
        fusion_method: str = "mean",
        distance_metric: str = "cosine",
        classifier_mode: str = "prototype",
        fusion_transformer_heads: int = 4,
        fusion_transformer_layers: int = 2,
        fusion_transformer_ffn_dim: int = 128,
        fusion_ib_beta: float = 1e-3,
    ):
        """
        Args:
            sequence_length: Length of temporal sequence
            num_sensors: Number of sensor channels
            num_classes: Number of task classes
            embedding_dim: Dimension of embedding space per modality
            modality_names: Names of modalities (EDA, ECG, EMG)
            filters_list: List of filters in each convolution layer
            strides: Strides of convolution layers
            fusion_method: 'mean', 'gated', or 'transformer_ib'
            distance_metric: 'euclidean' or 'cosine'
            num_tcn_blocks: number of Temporal Convolutional Network blocks
            tcn_dilation_rates: Dilation rate per TCN block
            tcn_kernel_size: Kernel size used by Conv1D layers inside each TCN block
            strides: Stride used by temporal pooling between TCN blocks
            pooling_size: Pool size used between TCN blocks
            tcn_dropout_rate: Dropout rate inside each TCN encoder
            tcn_attention_heads: Number of TCN self-attention heads
            tcn_attention_key_dim: Key dimension per TCN attention head
            tcn_attention_dropout: Dropout used by the TCN self-attention layer
            tcn_attention_pool_size: Downsample factor before TCN self-attention
            use_attention: If True, enable self-attention inside each TCN encoder
            fusion_transformer_heads: Number of attention heads in transformer fusion
            fusion_transformer_layers: Number of transformer blocks in fusion
            fusion_transformer_ffn_dim: FFN hidden size in transformer fusion
            fusion_ib_beta: KL regularization weight for information bottleneck
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.num_sensors = num_sensors
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.modality_names = modality_names
        self.fusion_method = fusion_method
        self.distance_metric = distance_metric
        self.classifier_mode = classifier_mode
        self.num_tcn_blocks = num_tcn_blocks
        self.tcn_dilation_rates = tcn_dilation_rates
        self.tcn_kernel_size = tcn_kernel_size
        self.strides = strides
        self.pooling_size = pooling_size
        self.filters_list = filters_list
        self.tcn_dropout_rate = tcn_dropout_rate
        self.tcn_attention_heads = tcn_attention_heads
        self.tcn_attention_key_dim = tcn_attention_key_dim
        self.tcn_attention_dropout = tcn_attention_dropout
        self.tcn_attention_pool_size = tcn_attention_pool_size
        self.use_attention = bool(use_attention)
        self.fusion_transformer_heads = fusion_transformer_heads
        self.fusion_transformer_layers = fusion_transformer_layers
        self.fusion_transformer_ffn_dim = fusion_transformer_ffn_dim
        self.fusion_ib_beta = fusion_ib_beta
        self.logit_scale = 10.0 if distance_metric == "cosine" else 1.0
        self.logger = setup_logger(name="MultimodalPrototypicalNetwork")
        # Create separate encoder for each modality
        self.modality_encoders = {}
        for modality_name in modality_names:
            self.modality_encoders[modality_name] = self._build_encoder(
                modality_name=modality_name,
                sequence_length=sequence_length,
                embedding_dim=embedding_dim,
                num_tcn_blocks=num_tcn_blocks,
                tcn_dilation_rates=tcn_dilation_rates,
                tcn_kernel_size=tcn_kernel_size,
                strides=strides,
                pooling_size=pooling_size,
                tcn_dropout_rate=tcn_dropout_rate,
                tcn_attention_heads=tcn_attention_heads,
                tcn_attention_key_dim=tcn_attention_key_dim,
                tcn_attention_dropout=tcn_attention_dropout,
                tcn_attention_pool_size=tcn_attention_pool_size,
                use_attention=use_attention,
                filters_list=filters_list,
            )

        # Fusion layer based on fusion method
        if fusion_method == "mean":
            self.fused_embedding_dim = embedding_dim
            self.fusion_layer = None
            self.gating_layer = None
            self.gating_norm_layers = None
            self.gating_softmax_layer = None
        elif fusion_method == "gated":
            self.fused_embedding_dim = embedding_dim
            self.fusion_layer = None
            self.gating_norm_layers = {
                modality_name: keras.layers.Dense(
                    embedding_dim,
                    activation="tanh",
                    name=f"gated_norm_{modality_name}",
                )
                for modality_name in modality_names
            }
            self.gating_softmax_layer = keras.layers.Dense(
                len(modality_names) * embedding_dim,
                name="gated_fusion_logits",
            )
            self.gating_layer = None
        elif fusion_method == "transformer_ib":
            self.fused_embedding_dim = embedding_dim
            self.fusion_layer = TransformerInformationBottleneckFusion(
                embedding_dim=embedding_dim,
                num_modalities=len(modality_names),
                num_heads=fusion_transformer_heads,
                num_layers=fusion_transformer_layers,
                ffn_dim=fusion_transformer_ffn_dim,
                ib_beta=fusion_ib_beta,
            )
            self.gating_norm_layers = None
            self.gating_softmax_layer = None
            self.gating_layer = None
            self.logger.debug("Build Fusion Model:")
            self.logger.debug(self.fusion_layer)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        self.logger.debug(
            f"Initialized MultimodalPrototypicalNetwork with {len(modality_names)} modalities"
        )
        self.logger.debug(
            f"Fusion method: {fusion_method}, Classifier mode: {classifier_mode}, "
            f"Final embedding dim: {self.fused_embedding_dim}"
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

    def _build_encoder(
        self,
        modality_name: str,
        sequence_length: int,
        embedding_dim: int,
        num_tcn_blocks: int,
        tcn_dilation_rates: Optional[List[int]],
        tcn_kernel_size: int,
        strides: int,
        pooling_size: int,
        tcn_dropout_rate: float,
        tcn_attention_heads: int,
        tcn_attention_key_dim: int,
        tcn_attention_dropout: float,
        tcn_attention_pool_size: int,
        use_attention: bool,
        filters_list: Optional[List[int]] = None,
    ) -> keras.models.Model:
        """Build 1D CNN encoder for a single modality."""
        model = TemporalConvolutionalNetwork(
            name=f"tcn_{modality_name}",
            sequence_length=sequence_length,
            embedding_dim=embedding_dim,
            num_blocks=num_tcn_blocks,
            dilation_rates=tcn_dilation_rates,
            kernel_size=tcn_kernel_size,
            dropout_rate=tcn_dropout_rate,
            num_attention_heads=tcn_attention_heads,
            attention_key_dim=tcn_attention_key_dim,
            attention_dropout=tcn_attention_dropout,
            strides=strides,
            pooling_size=pooling_size,
            attention_pool_size=tcn_attention_pool_size,
            use_attention=use_attention,
            filters_list=filters_list,
        )

        self.logger.debug(f"Built CNN encoder with {modality_name}")
        if self.logger.isEnabledFor(10):
            self.logger.debug("Encoder summary for %s:", modality_name)
            model.summary(print_fn=self.logger.debug)
        return model

    def _encode_modality_stack(self, x, training=False):
        """Encode each modality and return [batch, num_modalities, embedding_dim]."""
        modality_embeddings = []
        for i, modality_name in enumerate(self.modality_names):
            modality_data = x[:, :, i : i + 1]
            encoder = self.modality_encoders[modality_name]
            embedding = encoder(modality_data, training=training)
            modality_embeddings.append(embedding)

        return tf.stack(modality_embeddings, axis=1)

    def _fuse_modality_stack(self, fused, training=False):
        """Fuse [batch, num_modalities, embedding_dim] modality embeddings."""
        if self.fusion_method == "mean":
            fused = tf.reduce_mean(fused, axis=1)  # [batch, embedding_dim]
        elif self.fusion_method == "gated":
            normalized_embeddings = []
            modality_embeddings = tf.unstack(fused, axis=1)
            for modality_name, embedding in zip(
                self.modality_names, modality_embeddings
            ):
                normalized_embeddings.append(
                    self.gating_norm_layers[modality_name](embedding, training=training)
                )

            normalized_concat = tf.concat(
                normalized_embeddings, axis=1
            )  # [batch, num_modalities * embedding_dim]
            gate_logits = self.gating_softmax_layer(
                normalized_concat, training=training
            )
            gate_weights = tf.nn.softmax(gate_logits, axis=1)
            gate_weights = tf.reshape(
                gate_weights,
                [-1, len(self.modality_names), self.embedding_dim],
            )
            fused = tf.reduce_sum(fused * gate_weights, axis=1)
        else:  # self.fusion_method == "transformer_ib":
            fused = self.fusion_layer(fused, training=training)

        return fused

    def encode(self, x, training=False):
        """
        Map input to combined embedding space.

        Args:
            x: [batch_size, sequence_length, num_sensors]
            training: Whether in training mode

        Returns:
            embeddings: [batch_size, fused_embedding_dim] or [batch_size, embedding_dim]
        """
        return self._fuse_modality_stack(
            self._encode_modality_stack(x, training=training),
            training=training,
        )

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

    def compute_similarity_scores(self, query_embeddings, prototype_embeddings):
        """
        Compute similarity scores between query and class prototypes.

        Returns:
            similarities: [num_queries, num_classes]
        """
        if self.distance_metric == "cosine":
            query_norm = tf.nn.l2_normalize(query_embeddings, axis=1)
            prototype_norm = tf.nn.l2_normalize(prototype_embeddings, axis=1)
            return tf.matmul(query_norm, tf.transpose(prototype_norm))

        return -self.compute_distances(query_embeddings, prototype_embeddings)

    def compute_support_to_query_similarity(
        self, query_embeddings: tf.Tensor, support_embeddings: tf.Tensor
    ) -> tf.Tensor:
        """Compute query-to-support similarities."""
        if self.distance_metric == "cosine":
            query_norm = tf.nn.l2_normalize(query_embeddings, axis=1)
            support_norm = tf.nn.l2_normalize(support_embeddings, axis=1)
            return tf.matmul(query_norm, support_norm, transpose_b=True)

        return -self.compute_pairwise_distances(query_embeddings, support_embeddings)

    def compute_pairwise_distances(
        self, a_embeddings: tf.Tensor, b_embeddings: tf.Tensor
    ) -> tf.Tensor:
        """Compute pairwise distances between two embedding sets."""
        if self.distance_metric == "euclidean":
            return tf.sqrt(
                tf.reduce_sum(
                    (tf.expand_dims(a_embeddings, 1) - tf.expand_dims(b_embeddings, 0))
                    ** 2,
                    axis=2,
                )
                + 1e-8
            )
        if self.distance_metric == "cosine":
            a_norm = tf.nn.l2_normalize(a_embeddings, axis=1)
            b_norm = tf.nn.l2_normalize(b_embeddings, axis=1)
            return 1 - tf.matmul(a_norm, b_norm, transpose_b=True)
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

    def forward_episode(self, support_x, support_y, query_x, training=False):
        """Run one episode and return logits plus intermediate embedding tensors."""
        support_modality_embeddings = self._encode_modality_stack(
            support_x, training=training
        )
        query_modality_embeddings = self._encode_modality_stack(
            query_x, training=training
        )
        modality_mean = tf.reduce_mean(
            support_modality_embeddings, axis=0, keepdims=True
        )
        modality_std = (
            tf.math.reduce_std(support_modality_embeddings, axis=0, keepdims=True)
            + 1e-6
        )
        support_modality_embeddings = (
            support_modality_embeddings - modality_mean
        ) / modality_std
        query_modality_embeddings = (
            query_modality_embeddings - modality_mean
        ) / modality_std
        support_embeddings = self._fuse_modality_stack(
            support_modality_embeddings, training=training
        )
        query_embeddings = self._fuse_modality_stack(
            query_modality_embeddings, training=training
        )
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
