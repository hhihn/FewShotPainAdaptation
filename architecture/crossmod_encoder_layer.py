from tensorflow import keras


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
