from typing import List, Any
from keras import Model
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
        self.temporal_norm = keras.layers.LayerNormalization(name="temporal_norm")
        self.temporal_activation = keras.layers.Activation("elu", name="temporal_elu")

        self.depthwise_conv = keras.layers.DepthwiseConv2D(
            kernel_size=(1, self.num_sensors),
            depth_multiplier=self.depth_multiplier,
            use_bias=False,
            depthwise_constraint=keras.constraints.max_norm(1.0),
            name="sensor_depthwise_conv",
        )
        self.depthwise_norm = keras.layers.LayerNormalization(name="depthwise_norm")
        self.depthwise_activation = keras.layers.Activation(
            "elu", name="depthwise_elu"
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
        self.separable_norm = keras.layers.LayerNormalization(name="separable_norm")
        self.separable_activation = keras.layers.Activation(
            "elu", name="separable_elu"
        )
        self.pool_2 = keras.layers.AveragePooling2D(
            pool_size=(self.pool_size_2, 1),
            name="separable_average_pool",
        )
        self.dropout_2 = keras.layers.Dropout(
            rate=self.dropout_rate, name="separable_dropout"
        )
        self.global_pool = keras.layers.GlobalAveragePooling2D(name="global_pooling")
        self.embedding_dense = keras.layers.Dense(
            self.embedding_dim,
            activation=None,
            kernel_regularizer=keras.regularizers.l2(self.l2_weight),
            name="embedding_dense",
        )
        self.embedding_norm = keras.layers.LayerNormalization(name="embedding_norm")

        self.logger.debug(
            "Initialized EEGNetStyleEncoder with temporal_filters=%s, "
            "depth_multiplier=%s, separable_filters=%s, embedding_dim=%s",
            self.temporal_filters,
            self.depth_multiplier,
            self.separable_filters,
            self.embedding_dim,
        )

    def call(self, x, training=False):
        """Encode [batch, time, sensors] windows into [batch, embedding_dim]."""
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

        x = self.global_pool(x)
        x = self.embedding_dense(x)
        return self.embedding_norm(x, training=training)

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
            activation="elu",
            name="embedding_dense_hidden",
            kernel_initializer="he_normal",
        )
        self.embedding_dense_hidden_dropout = keras.layers.Dropout(
            rate=self.dropout_rate,
            name="embedding_dense_hidden_dropout",
        )
        self.embedding_dense_hidden_bn = keras.layers.LayerNormalization(
            name="embedding_dense_hidden_bn"
        )
        self.embedding_dense_hidden2 = keras.layers.Dense(
            512,
            activation="elu",
            name="embedding_dense_hidden",
            kernel_initializer="he_normal",
        )
        self.embedding_dense_hidden_dropout2 = keras.layers.Dropout(
            rate=self.dropout_rate,
            name="embedding_dense_hidden_dropout",
        )
        self.embedding_dense_hidden_bn2 = keras.layers.LayerNormalization(
            name="embedding_dense_hidden_bn"
        )
        self.embedding_dense = keras.layers.Dense(
            embedding_dim,
            activation="elu",
            name="embedding_dense",
            kernel_initializer="he_normal",
        )
        self.embedding_dense_bn = keras.layers.LayerNormalization(
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
            activation="elu",
            kernel_initializer="he_normal",
            name=f"cnn_block_{block_idx}_conv1",
        )(inputs)
        x = keras.layers.LayerNormalization(name=f"cnn_block_{block_idx}_ln1")(x)
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
