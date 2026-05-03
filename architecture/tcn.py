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
        transformer_layers: int = 4,
        transformer_ffn_dim: int = 256,
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
            attention_key_dim: Key dimension per attention head. For a Transformer
                model dimension of 64 with 8 heads, use 8.
            attention_dropout: Dropout in Transformer encoder layers
            transformer_layers: Number of Transformer encoder layers
            transformer_ffn_dim: Feed-forward hidden dimension in each Transformer layer
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
        self.num_attention_heads = num_attention_heads
        self.attention_key_dim = attention_key_dim
        self.attention_dropout = attention_dropout
        self.transformer_layers = transformer_layers
        self.transformer_ffn_dim = transformer_ffn_dim
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
            shape=(self.sequence_length, 1), name=f"tcn_block_{0}_input"
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
            self.attention_pool = None
            if self.attention_pool_size > 1:
                self.attention_pool = keras.layers.AveragePooling1D(
                    pool_size=self.attention_pool_size,
                    strides=self.attention_pool_size,
                    padding="valid",
                    name="attention_pool",
                )
            self.transformer_input_projection = keras.layers.Dense(
                embedding_dim,
                name="transformer_input_projection",
                kernel_initializer="he_normal",
            )
            self.transformer_attention_layers = []
            self.transformer_ffn_layers = []
            self.transformer_attention_norm_layers = []
            self.transformer_ffn_norm_layers = []
            self.transformer_attention_dropout_layers = []
            self.transformer_ffn_dropout_layers = []
            for layer_idx in range(transformer_layers):
                self.transformer_attention_layers.append(
                    keras.layers.MultiHeadAttention(
                        num_heads=num_attention_heads,
                        key_dim=attention_key_dim,
                        dropout=attention_dropout,
                        name=f"transformer_encoder_{layer_idx}_mha",
                        kernel_initializer="he_normal",
                    )
                )
                self.transformer_ffn_layers.append(
                    keras.Sequential(
                        [
                            keras.layers.Dense(
                                transformer_ffn_dim,
                                activation="relu",
                                kernel_initializer="he_normal",
                                name=f"transformer_encoder_{layer_idx}_ffn_dense",
                            ),
                            keras.layers.Dropout(
                                attention_dropout,
                                name=f"transformer_encoder_{layer_idx}_ffn_dropout",
                            ),
                            keras.layers.Dense(
                                embedding_dim,
                                kernel_initializer="he_normal",
                                name=f"transformer_encoder_{layer_idx}_ffn_output",
                            ),
                        ],
                        name=f"transformer_encoder_{layer_idx}_ffn",
                    )
                )
                self.transformer_attention_norm_layers.append(
                    keras.layers.LayerNormalization(
                        name=f"transformer_encoder_{layer_idx}_attn_norm"
                    )
                )
                self.transformer_ffn_norm_layers.append(
                    keras.layers.LayerNormalization(
                        name=f"transformer_encoder_{layer_idx}_ffn_norm"
                    )
                )
                self.transformer_attention_dropout_layers.append(
                    keras.layers.Dropout(
                        attention_dropout,
                        name=f"transformer_encoder_{layer_idx}_attn_dropout",
                    )
                )
                self.transformer_ffn_dropout_layers.append(
                    keras.layers.Dropout(
                        attention_dropout,
                        name=f"transformer_encoder_{layer_idx}_output_dropout",
                    )
                )
            self.global_pooling = keras.layers.GlobalAveragePooling1D(
                name="global_pooling"
            )

        else:
            # Flatten
            self.flatten_layer = keras.layers.Flatten(name="flatten")

        # Final embedding layers
        self.embedding_dense_hidden = keras.layers.Dense(
            256,
            activation="elu",
            name="embedding_dense_hidden",
            kernel_initializer="he_normal",
        )
        self.embedding_dense_hidden_dropout = keras.layers.Dropout(
            rate=self.dropout_rate,
            name="embedding_dense_hidden_dropout",
        )
        self.embedding_dense = keras.layers.Dense(
            embedding_dim,
            activation="elu",
            name="embedding_dense",
            kernel_initializer="he_normal",
        )
        self.embedding_dense_dropout = keras.layers.Dropout(
            rate=self.dropout_rate,
            name="embedding_dense_dropout",
        )
        self.embedding_norm = keras.layers.LayerNormalization(name="embedding_norm")

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
            x = self.transformer_input_projection(x)

            for layer_idx in range(self.transformer_layers):
                attention_output = self.transformer_attention_layers[layer_idx](
                    x, x, training=training
                )
                x = self.transformer_attention_norm_layers[layer_idx](
                    x
                    + self.transformer_attention_dropout_layers[layer_idx](
                        attention_output, training=training
                    )
                )
                ffn_output = self.transformer_ffn_layers[layer_idx](
                    x, training=training
                )
                x = self.transformer_ffn_norm_layers[layer_idx](
                    x
                    + self.transformer_ffn_dropout_layers[layer_idx](
                        ffn_output, training=training
                    )
                )
            if self.logger.isEnabledFor(10):
                self.logger.debug("After Transformer encoder: %s", x.shape)

            x = self.global_pooling(x)

        else:
            x = self.flatten_layer(x)

        if self.logger.isEnabledFor(10):
            self.logger.debug("After global pool: %s", x.shape)

        # Final embedding
        x = self.embedding_dense_hidden(x)
        x = self.embedding_dense_hidden_dropout(x, training=training)
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
            "num_attention_heads": self.num_attention_heads,
            "attention_key_dim": self.attention_key_dim,
            "attention_dropout": self.attention_dropout,
            "transformer_layers": self.transformer_layers,
            "transformer_ffn_dim": self.transformer_ffn_dim,
            "strides": self.strides,
            "pooling_size": self.pooling_size,
            "attention_pool_size": self.attention_pool_size,
            "use_attention": self.use_attention,
        }
