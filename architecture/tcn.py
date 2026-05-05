from typing import List, Any
import tensorflow as tf
from keras import Model
from tensorflow import keras

from utils.logger import setup_logger


class TemporalConvolutionalNetwork(keras.Model):
    """
    Modular Temporal Convolutional Network (TCN) with self-attention.

    Allows flexible configuration of:
    - Number of TCN blocks
    - Filter sizes per block
    - Dilation rates per block
    - Self-attention parameters
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
        num_attention_heads: int = 4,
        attention_key_dim: int = 32,
        attention_dropout: float = 0.2,
        strides: int = 2,
        pooling_size: int = 2,
        attention_pool_size: int = 8,
        use_attention: bool = False,
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
            num_attention_heads: Number of attention heads
            attention_key_dim: Key dimension per attention head
            attention_dropout: Dropout in attention layer
            strides: Stride used by temporal pooling between TCN blocks
            pooling_size: Pool size used between TCN blocks
            attention_pool_size: Downsampling factor before self-attention.
                Values > 1 reduce memory from O(L^2) to O((L/pool)^2).
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
        self.attention_pool_size = max(1, int(attention_pool_size))
        self.use_attention = use_attention
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
        self.tcn_blocks = []
        inputs = keras.layers.Input(
            shape=(self.sequence_length, 1), name=f"{0}_input"
        )
        for i in range(num_blocks):
            block, new_inputs = self._build_cnn_block(
                inputs=inputs,
                filters=filters_list[i],
                block_idx=i,
                pooling_stride=self.strides,
                pooling_size=self.pooling_size,
            )
            inputs = new_inputs
            self.tcn_blocks.append(block)

        if self.use_attention:
            # Self-attention layer
            self.attention = keras.layers.MultiHeadAttention(
                num_heads=num_attention_heads,
                key_dim=attention_key_dim,
                dropout=attention_dropout,
                name="self_attention",
                kernel_initializer="he_normal",
            )

            # Optional temporal downsampling before attention to avoid OOM on long sequences.
            self.attention_pool = None
            if self.attention_pool_size > 1:
                self.attention_pool = keras.layers.AveragePooling1D(
                    pool_size=self.attention_pool_size,
                    strides=self.attention_pool_size,
                    padding="valid",
                    name="attention_pool",
                )

            # Normalization after attention
            self.attention_norm = keras.layers.LayerNormalization(name="attention_norm")

            # Global pooling
            self.global_pooling = keras.layers.GlobalAveragePooling1D(
                name="global_pooling"
            )

        else:
            # Flatten
            self.flatten_layer = keras.layers.Flatten(name="flatten")

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
        self.embedding_dense_hidden_2 = keras.layers.Dense(
            512,
            activation="elu",
            name="embedding_dense_hidden_2",
            kernel_initializer="he_normal",
        )
        self.embedding_dense_hidden_dropout_2 = keras.layers.Dropout(
            rate=self.dropout_rate,
            name="embedding_dense_hidden_dropout_2",
        )
        self.embedding_dense = keras.layers.Dense(
            embedding_dim,
            activation="elu",
            name="embedding_dense",
            kernel_initializer="he_normal",
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
        x = keras.layers.BatchNormalization(name=f"cnn_block_{block_idx}_ln1")(x)
        x = keras.layers.MaxPool1D(
            pool_size=pooling_size,
            strides=pooling_stride,
            name=f"cnn_block_{block_idx}_maxpool",
        )(x)
        x = keras.layers.Dropout(rate=self.dropout_rate, name=f"cnn_block_{block_idx}_dropout")(x)
        return keras.Model(inputs=inputs, outputs=x, name=f"cnn_block_{block_idx}"), x

    def _build_tcn_block(
        self, inputs, filters: int, dilation_rate: int, block_idx: int
    ) -> tuple[Model, Any]:
        """Build a single TCN block with residual connection."""

        # Double convolution with normalization that is stable for small episodic batches.
        x = keras.layers.Conv1D(
            filters,
            kernel_size=self.kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            activation="elu",
            kernel_initializer="he_normal",
            name=f"tcn_block_{block_idx}_conv1",
        )(inputs)
        x = keras.layers.LayerNormalization(name=f"tcn_block_{block_idx}_ln1")(x)

        x = keras.layers.Conv1D(
            filters,
            kernel_size=self.kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            activation="elu",
            kernel_initializer="he_normal",
            name=f"tcn_block_{block_idx}_conv2",
        )(x)
        x = keras.layers.LayerNormalization(name=f"tcn_block_{block_idx}_ln2")(x)
        x = keras.layers.Dropout(
            self.dropout_rate, name=f"tcn_block_{block_idx}_dropout"
        )(x)

        # Residual connection with projection
        residual = keras.layers.Conv1D(
            filters,
            kernel_size=1,
            padding="same",
            kernel_initializer="he_normal",
            name=f"tcn_block_{block_idx}_residual_proj",
        )(inputs)
        residual_add = keras.layers.Add(name=f"tcn_block_{block_idx}_add")(
            [x, residual]
        )
        residual_norm = keras.layers.LayerNormalization(
            name=f"tcn_block_{block_idx}_resnorm"
        )(residual_add)
        outputs = keras.layers.MaxPool1D(
            pool_size=self.pooling_size,
            strides=self.strides,
            name=f"tcn_block_{block_idx}_pool",
        )(residual_norm)

        return keras.Model(
            inputs=inputs, outputs=outputs, name=f"tcn_block_{block_idx}"
        ), outputs

    def call(self, x, training=False):
        """
        Forward pass through TCN with attention.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, in_channels]
            training: Whether in training mode

        Returns:
            Embedding tensor of shape [batch_size, embedding_dim]
        """
        # Pass through TCN blocks sequentially
        for block in self.tcn_blocks:
            x = block(x, training=training)

        if self.logger.isEnabledFor(10):
            self.logger.debug("After TCN blocks: %s", x.shape)

        if self.use_attention:
            if self.attention_pool is not None:
                x = self.attention_pool(x)

            attention_output = self.attention(x, x, training=training)
            x = self.attention_norm(x + attention_output)
            if self.logger.isEnabledFor(10):
                self.logger.debug("After attention: %s", x.shape)

            # Global pooling
            x = self.global_pooling(x)
        else:
            x = self.flatten_layer(x)

        if self.logger.isEnabledFor(10):
            self.logger.debug("After global pool: %s", x.shape)

        # Final embedding
        x = self.embedding_dense_hidden(x)
        x = self.embedding_dense_hidden_dropout(x, training=training)
        x = self.embedding_dense_hidden_2(x)
        x = self.embedding_dense_hidden_dropout_2(x, training=training)
        x = self.embedding_dense(x)

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
            "attention_pool_size": self.attention_pool_size,
        }
