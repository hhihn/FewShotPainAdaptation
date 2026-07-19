import tensorflow as tf
from tensorflow import keras


class CrossAttentionModule(keras.layers.Layer):
    """Apply task-conditioned cross attention to temporal feature maps.

    The layer compares class prototype feature maps with query feature maps and
    returns pairwise similarity scores plus the intermediate attention tensors.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        meta_hidden_dim: int = 32,
        local_pool_temperature: float = 0.1,
        name: str = "cross_attention_module",
    ):
        """Initialize the cross-attention module.

        Args:
            temperature: Softmax temperature for temporal attention weights.
            meta_hidden_dim: Hidden width of the meta-kernel generator.
            local_pool_temperature: Temperature of smooth local temporal matching.
            name: Keras layer name.
        """
        super().__init__(name=name)
        self.temperature = float(temperature)
        self.meta_hidden_dim = int(meta_hidden_dim)
        self.local_pool_temperature = float(local_pool_temperature)
        if self.local_pool_temperature <= 0:
            raise ValueError("local_pool_temperature must be > 0")
        self._built_temporal_shapes = None

    def build(self, input_shape):
        """Create meta-kernel weights from static temporal feature shapes.

        Args:
            input_shape: Pair of prototype and query feature-map shapes.
        """
        prototype_shape, query_shape = input_shape
        if prototype_shape[-2] is None or query_shape[-2] is None:
            raise ValueError(
                "CAN requires statically known temporal feature-map length"
            )
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
        self.pool_gate_logit = self.add_weight(
            name="pool_gate_logit",
            shape=(),
            initializer="zeros",
            trainable=True,
        )
        self._built_temporal_shapes = (proto_time, query_time)
        super().build(input_shape)

    @staticmethod
    def _pair_descriptor(prototype_maps: tf.Tensor, query_maps: tf.Tensor) -> tf.Tensor:
        """Build pair descriptors for every query/prototype combination.

        Descriptors concatenate prototype summaries, query summaries, absolute
        differences, and elementwise products for the meta-kernel network.
        """
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
        """Generate prototype-side and query-side temporal kernels.

        The generated kernels are conditioned on each query/prototype pair and
        normalized over their corresponding temporal axes.
        """
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

    def _pool_temporal_features(
        self,
        feature_maps: tf.Tensor,
        attention: tf.Tensor,
        temporal_axis: int,
    ) -> tf.Tensor:
        """Mix true attention pooling with a stable global-mean fallback."""
        mean_descriptor = tf.reduce_mean(feature_maps, axis=temporal_axis)
        attention_descriptor = tf.reduce_sum(
            feature_maps * attention[..., tf.newaxis], axis=temporal_axis
        )
        gate = tf.sigmoid(tf.cast(self.pool_gate_logit, feature_maps.dtype))
        return (1.0 - gate) * mean_descriptor + gate * attention_descriptor

    def _smooth_local_pool(self, correlation: tf.Tensor) -> tf.Tensor:
        """Pool prototype-time matches without single-timestep max gradients."""
        temperature = tf.cast(self.local_pool_temperature, correlation.dtype)
        time_count = tf.cast(tf.shape(correlation)[3], correlation.dtype)
        return temperature * (
            tf.reduce_logsumexp(correlation / temperature, axis=3)
            - tf.math.log(time_count)
        )

    def call(self, inputs) -> dict[str, tf.Tensor]:
        """Return pairwise CAM logits and attention tensors.

        Args:
            inputs: `(prototype_maps, query_maps)`, where prototype maps are
                [tasks, classes, time, channels] and query maps are
                [tasks, queries, time, channels].

        Returns:
            Dictionary containing similarity scores, distances, temporal
            attentions, local logits, and the raw correlation tensor.
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

        pairwise_proto_maps = prototype_maps[:, tf.newaxis, :, :, :]
        pairwise_query_maps = query_maps[:, :, tf.newaxis, :, :]
        attended_proto_descriptors = self._pool_temporal_features(
            pairwise_proto_maps,
            proto_attention,
            temporal_axis=3,
        )
        attended_query_descriptors = self._pool_temporal_features(
            pairwise_query_maps,
            query_attention,
            temporal_axis=3,
        )
        pairwise_similarity = tf.reduce_sum(
            tf.nn.l2_normalize(attended_proto_descriptors, axis=-1)
            * tf.nn.l2_normalize(attended_query_descriptors, axis=-1),
            axis=-1,
        )
        local_logits = tf.transpose(self._smooth_local_pool(correlation), [0, 1, 3, 2])
        return {
            "similarity_scores": pairwise_similarity,
            "distances": 1.0 - pairwise_similarity,
            "proto_attention": proto_attention,
            "query_attention": query_attention,
            "local_logits": local_logits,
            "correlation": correlation,
        }
