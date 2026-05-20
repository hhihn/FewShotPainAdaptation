import math
from typing import List, Any
from keras import Model
import tensorflow as tf
from tensorflow import keras

from utils.logger import setup_logger


class EEGNetStyleEncoder(keras.Model):
    """Compact EEGNet-style encoder for joint multichannel physiological windows."""

    def __init__(
        self,
        sequence_length: int = 2500,
        num_sensors: int = 3,
        embedding_dim: int = 64,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        separable_filters: int = 16,
        temporal_kernel_size: int = 64,
        separable_kernel_size: int = 16,
        pool_size_1: int = 4,
        pool_size_2: int = 8,
        dropout_rate: float = 0.25,
        l2_weight: float = 1e-4,
        enable_embedding_projection: bool = True,
        name: str = "eegnet_encoder",
    ):
        super().__init__(name=name)
        self.sequence_length = int(sequence_length)
        self.num_sensors = int(num_sensors)
        self.embedding_dim = int(embedding_dim)
        self.temporal_filters = int(temporal_filters)
        self.depth_multiplier = int(depth_multiplier)
        self.separable_filters = int(separable_filters)
        self.temporal_kernel_size = int(temporal_kernel_size)
        self.separable_kernel_size = int(separable_kernel_size)
        self.pool_size_1 = int(pool_size_1)
        self.pool_size_2 = int(pool_size_2)
        self.dropout_rate = float(dropout_rate)
        self.l2_weight = float(l2_weight)
        self.enable_embedding_projection = bool(enable_embedding_projection)
        self.logger = setup_logger(name="EEGNetStyleEncoder")

        self.reshape = keras.layers.Reshape(
            (self.sequence_length, self.num_sensors, 1),
            name="eegnet_input_reshape",
        )
        self.temporal_conv = keras.layers.Conv2D(
            filters=self.temporal_filters,
            kernel_size=(self.temporal_kernel_size, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            name="temporal_conv",
        )
        self.temporal_norm = keras.layers.BatchNormalization(name="temporal_norm")
        self.temporal_activation = keras.layers.Activation("gelu", name="temporal_elu")
        self.depthwise_conv = keras.layers.DepthwiseConv2D(
            kernel_size=(1, self.num_sensors),
            depth_multiplier=self.depth_multiplier,
            use_bias=False,
            depthwise_constraint=keras.constraints.max_norm(1.0),
            name="sensor_depthwise_conv",
        )
        self.depthwise_norm = keras.layers.BatchNormalization(name="depthwise_norm")
        self.depthwise_activation = keras.layers.Activation(
            "gelu", name="depthwise_elu"
        )
        self.pool_1 = keras.layers.AveragePooling2D(
            pool_size=(self.pool_size_1, 1),
            name="depthwise_average_pool",
        )
        self.dropout_1 = keras.layers.Dropout(
            rate=self.dropout_rate, name="depthwise_dropout"
        )

        self.separable_conv = keras.layers.SeparableConv2D(
            filters=self.separable_filters,
            kernel_size=(self.separable_kernel_size, 1),
            padding="same",
            use_bias=False,
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
            name="separable_temporal_conv",
        )
        self.separable_norm = keras.layers.BatchNormalization(name="separable_norm")
        self.separable_activation = keras.layers.Activation(
            "gelu", name="separable_elu"
        )
        self.pool_2 = keras.layers.AveragePooling2D(
            pool_size=(self.pool_size_2, 1),
            name="separable_average_pool",
        )
        self.dropout_2 = keras.layers.Dropout(
            rate=self.dropout_rate, name="separable_dropout"
        )
        if self.enable_embedding_projection:
            self.global_pool = keras.layers.GlobalAveragePooling2D(
                name="global_pooling"
            )
            self.embedding_dense = keras.layers.Dense(
                self.embedding_dim,
                activation=None,
                kernel_regularizer=keras.regularizers.l2(self.l2_weight),
                name="embedding_dense",
            )
            self.embedding_norm = keras.layers.BatchNormalization(name="embedding_norm")
        else:
            self.global_pool = None
            self.embedding_dense = None
            self.embedding_norm = None

        self.logger.debug(
            "Initialized EEGNetStyleEncoder with temporal_filters=%s, "
            "depth_multiplier=%s, separable_filters=%s, embedding_projection=%s",
            self.temporal_filters,
            self.depth_multiplier,
            self.separable_filters,
            self.enable_embedding_projection,
        )

    def _feature_map_4d(self, x, training=False):
        """Return post-separable EEGNet activations as [batch, time, 1, channels]."""
        x = self.reshape(x)
        x = self.temporal_conv(x)
        x = self.temporal_norm(x, training=training)
        x = self.temporal_activation(x)

        x = self.depthwise_conv(x)
        x = self.depthwise_norm(x, training=training)
        x = self.depthwise_activation(x)
        x = self.pool_1(x)
        x = self.dropout_1(x, training=training)

        x = self.separable_conv(x)
        x = self.separable_norm(x, training=training)
        x = self.separable_activation(x)
        x = self.pool_2(x)
        x = self.dropout_2(x, training=training)
        return x

    def extract_feature_map(self, x, training=False):
        """Return post-sensor-mixing temporal features as [batch, time, channels]."""
        return self._feature_map_4d(x, training=training)[:, :, 0, :]

    def embed_feature_map(self, feature_map, training=False):
        """Pool/project a temporal feature map into the configured embedding space."""
        if not self.enable_embedding_projection:
            raise RuntimeError("Embedding projection is disabled for this encoder.")
        x = feature_map
        if len(x.shape) == 3:
            x = x[:, :, tf.newaxis, :]
        x = self.global_pool(x)
        x = self.embedding_dense(x)
        return self.embedding_norm(x, training=training)

    def call(self, x, training=False):
        """Encode [batch, time, sensors] windows into [batch, embedding_dim]."""
        if not self.enable_embedding_projection:
            raise RuntimeError("Embedding projection is disabled for this encoder.")
        return self.embed_feature_map(
            self._feature_map_4d(x, training=training),
            training=training,
        )

    def get_config(self):
        """Return model configuration for serialization."""
        return {
            "sequence_length": self.sequence_length,
            "num_sensors": self.num_sensors,
            "embedding_dim": self.embedding_dim,
            "temporal_filters": self.temporal_filters,
            "depth_multiplier": self.depth_multiplier,
            "separable_filters": self.separable_filters,
            "temporal_kernel_size": self.temporal_kernel_size,
            "separable_kernel_size": self.separable_kernel_size,
            "pool_size_1": self.pool_size_1,
            "pool_size_2": self.pool_size_2,
            "dropout_rate": self.dropout_rate,
            "l2_weight": self.l2_weight,
            "enable_embedding_projection": self.enable_embedding_projection,
        }


class FourierPositionalEncoding1D(keras.layers.Layer):
    """Fourier positional encoding for temporal feature maps."""

    def __init__(
        self,
        d_model: int,
        base: float = 10000.0,
        name: str = "fourier_positional_encoding",
    ):
        super().__init__(name=name)
        self.d_model = int(d_model)
        self.base = float(base)
        if self.d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.base <= 0:
            raise ValueError("base must be > 0")

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.cast(tf.range(seq_len)[:, tf.newaxis], x.dtype)
        div_term = tf.exp(
            tf.cast(tf.range(0, self.d_model, 2), x.dtype)
            * tf.cast(-(math.log(self.base) / self.d_model), x.dtype)
        )
        sin_values = tf.sin(positions * div_term)
        cos_values = tf.cos(positions * div_term)
        pe = tf.reshape(
            tf.stack([sin_values, cos_values], axis=-1),
            [seq_len, -1],
        )[:, : self.d_model]
        return x + pe[tf.newaxis, :, :]


class CrossModTransformerEncoderLayer(keras.layers.Layer):
    """Pre-norm Transformer encoder layer for one physiological modality."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout_rate: float,
        ff_activation: str,
        name: str,
    ):
        super().__init__(name=name)
        self.input_dim = int(input_dim)
        self.num_heads = int(num_heads)
        self.hidden_dim = int(hidden_dim)
        self.dropout_rate = float(dropout_rate)
        self.ff_activation = str(ff_activation)
        if self.input_dim % self.num_heads != 0:
            raise ValueError("CrossMod attention dims must be divisible by num_heads")

        self.attention_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_attention_norm"
        )
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.input_dim // self.num_heads,
            dropout=self.dropout_rate,
            name=f"{name}_self_attention",
        )
        self.attention_dropout = keras.layers.Dropout(
            self.dropout_rate, name=f"{name}_attention_dropout"
        )
        self.ff_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name=f"{name}_ff_norm"
        )
        self.ff_dense_1 = keras.layers.Dense(
            self.hidden_dim,
            activation=self.ff_activation,
            name=f"{name}_ff_dense_1",
        )
        self.ff_dropout = keras.layers.Dropout(
            self.dropout_rate, name=f"{name}_ff_dropout"
        )
        self.ff_dense_2 = keras.layers.Dense(
            self.input_dim,
            activation=None,
            name=f"{name}_ff_dense_2",
        )

    def call(self, x, training=False):
        attention_input = self.attention_norm(x)
        attention_output = self.attention(
            attention_input,
            attention_input,
            training=training,
        )
        x = x + self.attention_dropout(attention_output, training=training)

        ff_input = self.ff_norm(x)
        ff_output = self.ff_dense_1(ff_input)
        ff_output = self.ff_dropout(ff_output, training=training)
        ff_output = self.ff_dense_2(ff_output)
        return x + ff_output


class CrossModFeatureMapEncoder(keras.Model):
    """EDA/ECG EEGNet-1D frontends fused by CrossMod self/cross attention."""

    def __init__(
        self,
        sequence_length: int = 2500,
        num_sensors: int = 2,
        frontend_temporal_filters: int = 8,
        frontend_separable_filters: int = 16,
        frontend_temporal_kernel_size: int = 64,
        frontend_separable_kernel_size: int = 16,
        frontend_pool_size_1: int = 4,
        frontend_pool_size_2: int = 8,
        frontend_dropout_rate: float = 0.25,
        frontend_l2_weight: float = 1e-4,
        num_heads: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        positional_base: float = 10000.0,
        attention_dropout_rate: float = 0.0,
        ff_activation: str = "relu",
        name: str = "crossmod_encoder",
    ):
        super().__init__(name=name)
        self.sequence_length = int(sequence_length)
        self.num_sensors = int(num_sensors)
        self.frontend_temporal_filters = int(frontend_temporal_filters)
        self.frontend_separable_filters = int(frontend_separable_filters)
        self.frontend_temporal_kernel_size = int(frontend_temporal_kernel_size)
        self.frontend_separable_kernel_size = int(frontend_separable_kernel_size)
        self.frontend_pool_size_1 = int(frontend_pool_size_1)
        self.frontend_pool_size_2 = int(frontend_pool_size_2)
        self.frontend_dropout_rate = float(frontend_dropout_rate)
        self.frontend_l2_weight = float(frontend_l2_weight)
        self.num_heads = int(num_heads)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.positional_base = float(positional_base)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.ff_activation = str(ff_activation)
        self.enable_embedding_projection = False
        self.global_pool = None
        self.embedding_dense = None
        self.embedding_norm = None

        self._validate_config()
        regularizer = (
            keras.regularizers.l2(self.frontend_l2_weight)
            if self.frontend_l2_weight > 0
            else None
        )

        self.eda_branch = self._build_frontend_branch(
            "eda", regularizer=regularizer
        )
        self.ecg_branch = self._build_frontend_branch(
            "ecg", regularizer=regularizer
        )
        self.eda_position = FourierPositionalEncoding1D(
            self.frontend_separable_filters,
            base=self.positional_base,
            name="crossmod_eda_positional_encoding",
        )
        self.ecg_position = FourierPositionalEncoding1D(
            self.frontend_separable_filters,
            base=self.positional_base,
            name="crossmod_ecg_positional_encoding",
        )
        self.eda_transformer_layers = [
            CrossModTransformerEncoderLayer(
                input_dim=self.frontend_separable_filters,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                dropout_rate=self.attention_dropout_rate,
                ff_activation=self.ff_activation,
                name=f"crossmod_eda_transformer_{idx}",
            )
            for idx in range(self.num_layers)
        ]
        self.ecg_transformer_layers = [
            CrossModTransformerEncoderLayer(
                input_dim=self.frontend_separable_filters,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                dropout_rate=self.attention_dropout_rate,
                ff_activation=self.ff_activation,
                name=f"crossmod_ecg_transformer_{idx}",
            )
            for idx in range(self.num_layers)
        ]
        self.eda_to_ecg_projection = keras.layers.Dense(
            self.frontend_separable_filters,
            name="crossmod_eda_to_ecg_projection",
        )
        self.eda_to_ecg_attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.frontend_separable_filters // self.num_heads,
            dropout=self.attention_dropout_rate,
            name="crossmod_eda_to_ecg_attention",
        )
        self.ecg_to_eda_attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.frontend_separable_filters // self.num_heads,
            dropout=self.attention_dropout_rate,
            name="crossmod_ecg_to_eda_attention",
        )

    def _validate_config(self) -> None:
        positive_int_fields = {
            "sequence_length": self.sequence_length,
            "num_sensors": self.num_sensors,
            "frontend_temporal_filters": self.frontend_temporal_filters,
            "frontend_separable_filters": self.frontend_separable_filters,
            "frontend_temporal_kernel_size": self.frontend_temporal_kernel_size,
            "frontend_separable_kernel_size": self.frontend_separable_kernel_size,
            "frontend_pool_size_1": self.frontend_pool_size_1,
            "frontend_pool_size_2": self.frontend_pool_size_2,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }
        for name, value in positive_int_fields.items():
            if value <= 0:
                raise ValueError(f"crossmod_{name} must be > 0")
        if self.num_sensors != 2:
            raise ValueError("CrossModFeatureMapEncoder requires num_sensors=2")
        if self.frontend_dropout_rate < 0 or self.frontend_dropout_rate >= 1:
            raise ValueError("crossmod_frontend_dropout_rate must be in [0, 1)")
        if self.attention_dropout_rate < 0 or self.attention_dropout_rate >= 1:
            raise ValueError("crossmod_attention_dropout_rate must be in [0, 1)")
        if self.frontend_l2_weight < 0:
            raise ValueError("crossmod_frontend_l2_weight must be non-negative")
        if self.positional_base <= 0:
            raise ValueError("crossmod_positional_base must be > 0")
        if self.frontend_separable_filters % self.num_heads != 0:
            raise ValueError(
                "crossmod_frontend_separable_filters must be divisible by crossmod_num_heads"
            )

    def _build_frontend_branch(self, prefix: str, regularizer):
        return keras.Sequential(
            [
                keras.layers.Conv1D(
                    self.frontend_temporal_filters,
                    self.frontend_temporal_kernel_size,
                    padding="same",
                    use_bias=False,
                    kernel_initializer="he_normal",
                    kernel_regularizer=regularizer,
                    name=f"{prefix}_frontend_temporal_conv",
                ),
                keras.layers.BatchNormalization(name=f"{prefix}_frontend_temporal_norm"),
                keras.layers.Activation("gelu", name=f"{prefix}_frontend_temporal_gelu"),
                keras.layers.AveragePooling1D(
                    pool_size=self.frontend_pool_size_1,
                    name=f"{prefix}_frontend_pool_1",
                ),
                keras.layers.Dropout(
                    self.frontend_dropout_rate,
                    name=f"{prefix}_frontend_dropout_1",
                ),
                keras.layers.SeparableConv1D(
                    self.frontend_separable_filters,
                    self.frontend_separable_kernel_size,
                    padding="same",
                    use_bias=False,
                    depthwise_initializer="he_normal",
                    pointwise_initializer="he_normal",
                    depthwise_regularizer=regularizer,
                    pointwise_regularizer=regularizer,
                    name=f"{prefix}_frontend_separable_conv",
                ),
                keras.layers.BatchNormalization(name=f"{prefix}_frontend_separable_norm"),
                keras.layers.Activation("gelu", name=f"{prefix}_frontend_separable_gelu"),
                keras.layers.AveragePooling1D(
                    pool_size=self.frontend_pool_size_2,
                    name=f"{prefix}_frontend_pool_2",
                ),
                keras.layers.Dropout(
                    self.frontend_dropout_rate,
                    name=f"{prefix}_frontend_dropout_2",
                ),
            ],
            name=f"{prefix}_eegnet_1d_frontend",
        )

    def extract_modality_feature_maps(self, x, training=False):
        if x.shape.rank != 3:
            raise ValueError("CrossMod input must have shape [batch, time, sensors]")
        if x.shape[-1] is not None and int(x.shape[-1]) != 2:
            raise ValueError("CrossMod input must contain exactly EDA and ECG channels")
        eda = x[:, :, 0:1]
        ecg = x[:, :, 1:2]
        eda_features = self.eda_branch(eda, training=training)
        ecg_features = self.ecg_branch(ecg, training=training)
        tf.debugging.assert_equal(
            tf.shape(eda_features)[1],
            tf.shape(ecg_features)[1],
            message="EDA and ECG frontend temporal lengths must match",
        )
        return eda_features, ecg_features

    def extract_feature_map(self, x, training=False):
        eda_features, ecg_features = self.extract_modality_feature_maps(
            x, training=training
        )
        eda_encoded = self.eda_position(eda_features)
        ecg_encoded = self.ecg_position(ecg_features)
        for layer in self.eda_transformer_layers:
            eda_encoded = layer(eda_encoded, training=training)
        for layer in self.ecg_transformer_layers:
            ecg_encoded = layer(ecg_encoded, training=training)

        eda_projected = self.eda_to_ecg_projection(eda_encoded)
        eda_to_ecg = self.eda_to_ecg_attention(
            query=eda_projected,
            value=ecg_encoded,
            key=ecg_encoded,
            training=training,
        )
        ecg_to_eda = self.ecg_to_eda_attention(
            query=ecg_encoded,
            value=eda_projected,
            key=eda_projected,
            training=training,
        )
        return tf.concat([eda_to_ecg, ecg_to_eda], axis=-1)

    def call(self, x, training=False):
        """CrossMod is a representation encoder and returns fused feature maps."""
        return self.extract_feature_map(x, training=training)

    def embed_feature_map(self, feature_map, training=False):
        del training
        raise RuntimeError("CrossModFeatureMapEncoder does not produce embeddings.")

    def get_config(self):
        return {
            "sequence_length": self.sequence_length,
            "num_sensors": self.num_sensors,
            "frontend_temporal_filters": self.frontend_temporal_filters,
            "frontend_separable_filters": self.frontend_separable_filters,
            "frontend_temporal_kernel_size": self.frontend_temporal_kernel_size,
            "frontend_separable_kernel_size": self.frontend_separable_kernel_size,
            "frontend_pool_size_1": self.frontend_pool_size_1,
            "frontend_pool_size_2": self.frontend_pool_size_2,
            "frontend_dropout_rate": self.frontend_dropout_rate,
            "frontend_l2_weight": self.frontend_l2_weight,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "positional_base": self.positional_base,
            "attention_dropout_rate": self.attention_dropout_rate,
            "ff_activation": self.ff_activation,
        }


class ConvolutionalNetwork(keras.Model):
    """
    Modular Convolutional Network

    Allows flexible configuration of:
    - Number of TCN blocks
    - Filter sizes per block
    - Dilation rates per block
    """

    def __init__(
        self,
        sequence_length: int = 2500,
        embedding_dim: int = 64,
        num_blocks: int = 3,
        filters_list: List[int] = None,
        dilation_rates: List[int] = None,
        kernel_size: int = 3,
        dropout_rate: float = 0.3,
        strides: int = 2,
        pooling_size: int = 2,
        name: str = "tcn_network",
    ):
        """
        Args:
            sequence_length: Length of input sequence
            embedding_dim: Final embedding dimension
            num_blocks: Number of TCN blocks
            filters_list: List of filter sizes per block. Auto-generated if None.
            dilation_rates: List of dilation rates per block. Auto-generated if None.
            kernel_size: Kernel size for all convolutions
            dropout_rate: Dropout rate in TCN blocks
            strides: Stride used by temporal pooling between TCN blocks
            pooling_size: Pool size used between TCN blocks
            name: Name of the model
        """
        super().__init__(name=name)

        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.strides = strides
        self.pooling_size = pooling_size
        self.logger = setup_logger(name="TemporalConvolutionalNetwork")
        # Auto-generate filter list if not provided
        if filters_list is None:
            filters_list = [32 * (2**i) for i in range(num_blocks)]

        # Auto-generate dilation rates if not provided
        if dilation_rates is None:
            dilation_rates = [2**i for i in range(num_blocks)]

        assert len(filters_list) == num_blocks, (
            "filters_list length must match num_blocks"
        )
        assert len(dilation_rates) == num_blocks, (
            "dilation_rates length must match num_blocks"
        )

        self.filters_list = filters_list
        self.dilation_rates = dilation_rates

        # Build TCN blocks
        self.blocks = []
        inputs = keras.layers.Input(shape=(self.sequence_length, 1), name=f"{0}_input")
        for i in range(num_blocks):
            block, new_inputs = self._build_cnn_block(
                inputs=inputs,
                filters=filters_list[i],
                block_idx=i,
                pooling_stride=self.strides,
                pooling_size=self.pooling_size,
            )
            inputs = new_inputs
            self.blocks.append(block)

        self.flatten_layer = keras.layers.Flatten()

        # Final embedding layers
        self.embedding_dense_hidden = keras.layers.Dense(
            1024,
            activation="gelu",
            name="embedding_dense_hidden",
            kernel_initializer="he_normal",
        )
        self.embedding_dense_hidden_dropout = keras.layers.Dropout(
            rate=self.dropout_rate,
            name="embedding_dense_hidden_dropout",
        )
        self.embedding_dense_hidden_bn = keras.layers.BatchNormalization(
            name="embedding_dense_hidden_bn"
        )
        self.embedding_dense_hidden2 = keras.layers.Dense(
            512,
            activation="gelu",
            name="embedding_dense_hidden",
            kernel_initializer="he_normal",
        )
        self.embedding_dense_hidden_dropout2 = keras.layers.Dropout(
            rate=self.dropout_rate,
            name="embedding_dense_hidden_dropout",
        )
        self.embedding_dense_hidden_bn2 = keras.layers.BatchNormalization(
            name="embedding_dense_hidden_bn"
        )
        self.embedding_dense = keras.layers.Dense(
            embedding_dim,
            activation="gelu",
            name="embedding_dense",
            kernel_initializer="he_normal",
        )
        self.embedding_dense_bn = keras.layers.BatchNormalization(
            name="embedding_dense_bn"
        )

        self.logger.debug(f"Initialized TCN with {num_blocks} blocks")
        self.logger.debug(f"Filters: {filters_list}")
        self.logger.debug(f"Dilation rates: {dilation_rates}")

    def build(self, input_shape):
        """Mark the subclassed model as buildable; child layers build on first call."""
        super().build(input_shape)

    def _build_cnn_block(
        self, inputs, filters: int, pooling_size: int, pooling_stride, block_idx: int
    ) -> tuple[Model, Any]:
        """Build a single CNN block."""
        x = keras.layers.Conv1D(
            filters,
            kernel_size=self.kernel_size,
            strides=1,
            activation="gelu",
            kernel_initializer="he_normal",
            name=f"cnn_block_{block_idx}_conv1",
        )(inputs)
        x = keras.layers.BatchNormalization(name=f"cnn_block_{block_idx}_ln1")(x)
        x = keras.layers.MaxPool1D(
            pool_size=pooling_size,
            strides=pooling_stride,
            name=f"cnn_block_{block_idx}_maxpool",
        )(x)
        x = keras.layers.Dropout(
            rate=self.dropout_rate, name=f"cnn_block_{block_idx}_dropout"
        )(x)
        return keras.Model(inputs=inputs, outputs=x, name=f"cnn_block_{block_idx}"), x

    def call(self, x, training=False):
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, in_channels]
            training: Whether in training mode

        Returns:
            Embedding tensor of shape [batch_size, embedding_dim]
        """
        # Pass through blocks sequentially
        for block in self.blocks:
            x = block(x, training=training)

        if self.logger.isEnabledFor(10):
            self.logger.debug("After blocks: %s", x.shape)

        x = self.flatten_layer(x)

        if self.logger.isEnabledFor(10):
            self.logger.debug("After global pool: %s", x.shape)

        # Final embedding
        x = self.embedding_dense_hidden(x)
        x = self.embedding_dense_hidden_bn(x, training=training)
        x = self.embedding_dense_hidden_dropout(x, training=training)
        x = self.embedding_dense_hidden2(x)
        x = self.embedding_dense_hidden_bn2(x, training=training)
        x = self.embedding_dense_hidden_dropout2(x, training=training)
        x = self.embedding_dense(x)
        x = self.embedding_dense_bn(x, training=training)

        return x

    def get_config(self):
        """Return model configuration for serialization."""
        return {
            "sequence_length": self.sequence_length,
            "embedding_dim": self.embedding_dim,
            "num_blocks": self.num_blocks,
            "filters_list": self.filters_list,
            "dilation_rates": self.dilation_rates,
            "kernel_size": self.kernel_size,
            "dropout_rate": self.dropout_rate,
            "strides": self.strides,
            "pooling_size": self.pooling_size,
        }
