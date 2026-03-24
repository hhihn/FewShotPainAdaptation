from typing import List, Any
import tensorflow as tf
from keras import Model
from tensorflow import keras

from utils.logger import setup_logger


class TemporalConvolutionalNetwork(keras.Model):
    """
    Modular Temporal Convolutional Network (cnn) with self-attention.

    Allows flexible configuration of:
    - Number of cnn blocks
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
        strides: int = 2,
        pooling_size: int = 2,
        dropout_rate: float = 0.3,
        num_attention_heads: int = 4,
        attention_key_dim: int = 32,
        attention_dropout: float = 0.2,
        attention_pool_size: int = 8,
        name: str = "cnn_network",
    ):
        """
        Args:
            sequence_length: Length of input sequence
            embedding_dim: Final embedding dimension
            num_blocks: Number of cnn blocks
            strides: Stride in convolution layer
            filters_list: List of filter sizes per block. Auto-generated if None.
            pooling_size: Pooling size per block. Auto-generated if None.
            kernel_size: Kernel size for all convolutions
            dropout_rate: Dropout rate in cnn blocks
            num_attention_heads: Number of attention heads
            attention_key_dim: Key dimension per attention head
            attention_dropout: Dropout in attention layer
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
        self.logger = setup_logger(name="TemporalConvolutionalNetwork")
        # Auto-generate filter list if not provided
        if filters_list is None:
            filters_list = [32 * (2**i) for i in range(num_blocks)]

        # Auto-generate dilation rates if not provided
        if self.pooling_size is None:
            self.pooling_size = 3

        if self.strides is None:
            self.strides = 2

        assert len(filters_list) == num_blocks, (
            "filters_list length must match num_blocks"
        )

        self.filters_list = filters_list

        # Build cnn blocks
        self.cnn_blocks = []
        inputs = keras.layers.Input(
            shape=(self.sequence_length, 1), name=f"cnn_block_{0}_input"
        )
        for i in range(num_blocks):
            block, new_inputs = self._build_cnn_block(
                inputs=inputs,
                filters=filters_list[i],
                pooling_size=self.pooling_size,
                strides=strides,
                block_idx=i,
            )
            inputs = new_inputs
            self.cnn_blocks.append(block)

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
        self.global_pool = keras.layers.GlobalAveragePooling1D(name="global_pool")

        # Final embedding layers
        self.embedding_dense = keras.layers.Dense(
            embedding_dim,
            activation="relu",
            name="embedding_dense",
            kernel_initializer="he_normal",
        )
        self.embedding_norm = keras.layers.LayerNormalization(name="embedding_norm")

        self.logger.debug(f"Initialized cnn with {num_blocks} blocks")
        self.logger.debug(f"Filters: {filters_list}")
        self.logger.debug(f"Dilation rates: {dilation_rates}")

    def _build_cnn_block(
        self, inputs, filters: int, strides: int, pooling_size: int, block_idx: int
    ) -> tuple[Model, Any]:
        """Build a single cnn block with residual connection."""

        # Double convolution with batch norm
        x = keras.layers.Conv1D(
            filters,
            kernel_size=self.kernel_size,
            strides=strides,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name=f"cnn_block_{block_idx}_conv1",
        )(inputs)
        x = keras.layers.LayerNormalization(name=f"cnn_block_{block_idx}_bn1")(x)
        x = keras.layers.Conv1D(
            filters,
            kernel_size=self.kernel_size,
            strides=strides,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name=f"cnn_block_{block_idx}_conv2",
        )(x)
        x = keras.layers.LayerNormalization(name=f"cnn_block_{block_idx}_bn2")(x)
        x = keras.layers.Dropout(
            self.dropout_rate, name=f"cnn_block_{block_idx}_dropout"
        )(x)
        outputs = keras.layers.AveragePooling1D(pool_size=pooling_size)(x)

        return keras.Model(
            inputs=inputs, outputs=outputs, name=f"cnn_block_{block_idx}"
        ), outputs

    def call(self, x, training=False):
        """
        Forward pass through cnn with attention.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, in_channels]
            training: Whether in training mode

        Returns:
            Embedding tensor of shape [batch_size, embedding_dim]
        """
        # Pass through cnn blocks sequentially
        for block in self.cnn_blocks:
            x = block(x, training=training)

        self.logger.debug(f"After cnn blocks: {tf.shape(x)}")

        # Self-attention
        attention_x = x
        if self.attention_pool is not None:
            attention_x = self.attention_pool(attention_x)

        attention_out = self.attention(attention_x, attention_x, training=training)
        x = self.attention_norm(attention_x + attention_out)

        self.logger.debug(f"After attention: {tf.shape(x)}")

        # Global pooling
        x = self.global_pool(x)

        self.logger.debug(f"After global pool: {tf.shape(x)}")

        # Final embedding
        x = self.embedding_dense(x)
        x = self.embedding_norm(x)

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
            "attention_pool_size": self.attention_pool_size,
        }
