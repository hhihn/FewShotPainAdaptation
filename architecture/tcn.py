from typing import List
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging

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
            name: Name of the model
        """
        super().__init__(name=name)

        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.logger = setup_logger(name="name", level=logging.INFO)
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
        for i in range(num_blocks):
            block = self._build_tcn_block(
                filters=filters_list[i], dilation_rate=dilation_rates[i], block_idx=i
            )
            self.tcn_blocks.append(block)

        # Self-attention layer
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=attention_key_dim,
            dropout=attention_dropout,
            name="self_attention",
        )

        # Normalization after attention
        self.attention_norm = keras.layers.LayerNormalization(name="attention_norm")

        # Global pooling
        self.global_pool = keras.layers.GlobalAveragePooling1D(name="global_pool")

        # Final embedding layers
        self.embedding_dense = keras.layers.Dense(
            embedding_dim, activation="relu", name="embedding_dense"
        )
        self.embedding_norm = keras.layers.LayerNormalization(name="embedding_norm")

        self.logger.info(f"Initialized TCN with {num_blocks} blocks")
        self.logger.info(f"Filters: {filters_list}")
        self.logger.info(f"Dilation rates: {dilation_rates}")

    def _build_tcn_block(
        self, filters: int, dilation_rate: int, block_idx: int
    ) -> keras.Model:
        """Build a single TCN block with residual connection."""
        inputs = keras.layers.Input(
            shape=(self.sequence_length, None), name=f"tcn_block_{block_idx}_input"
        )

        # Double convolution with batch norm
        x = keras.layers.Conv1D(
            filters,
            kernel_size=self.kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            activation="relu",
            name=f"tcn_block_{block_idx}_conv1",
        )(inputs)
        x = keras.layers.BatchNormalization(name=f"tcn_block_{block_idx}_bn1")(x)

        x = keras.layers.Conv1D(
            filters,
            kernel_size=self.kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            activation="relu",
            name=f"tcn_block_{block_idx}_conv2",
        )(x)
        x = keras.layers.BatchNormalization(name=f"tcn_block_{block_idx}_bn2")(x)
        x = keras.layers.Dropout(
            self.dropout_rate, name=f"tcn_block_{block_idx}_dropout"
        )(x)

        # Residual connection with projection
        residual = keras.layers.Conv1D(
            filters,
            kernel_size=1,
            padding="same",
            name=f"tcn_block_{block_idx}_residual_proj",
        )(inputs)

        outputs = keras.layers.Add(name=f"tcn_block_{block_idx}_add")([x, residual])

        return keras.Model(
            inputs=inputs, outputs=outputs, name=f"tcn_block_{block_idx}"
        )

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

        self.logger.debug(f"After TCN blocks: {tf.shape(x)}")

        # Self-attention
        attention_out = self.attention(x, x, training=training)
        x = self.attention_norm(x + attention_out)

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
        }
