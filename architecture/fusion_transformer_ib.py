import tensorflow as tf
from tensorflow import keras


class TransformerInformationBottleneckFusion(keras.layers.Layer):
    """Fuse modality embeddings with a Transformer and IB regularization."""

    def __init__(
        self,
        embedding_dim: int,
        num_modalities: int,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 128,
        dropout_rate: float = 0.1,
        ib_beta: float = 1e-3,
        name: str = "transformer_ib_fusion",
    ):
        super().__init__(name=name)
        self.embedding_dim = embedding_dim
        self.num_modalities = num_modalities
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate
        self.ib_beta = ib_beta
        self.last_kl = tf.constant(0.0, dtype=tf.float32)

        self.positional_embedding = self.add_weight(
            name="positional_embedding",
            shape=(1, num_modalities, embedding_dim),
            initializer="zeros",
            trainable=True,
        )

        self.attn_layers = []
        self.ffn_layers = []
        self.norm1_layers = []
        self.norm2_layers = []
        self.dropout_layers = []
        for _ in range(num_layers):
            self.attn_layers.append(
                keras.layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=max(1, embedding_dim // num_heads),
                    dropout=dropout_rate,
                )
            )
            self.ffn_layers.append(
                keras.Sequential(
                    [
                        keras.layers.Dense(ffn_dim, activation="relu"),
                        keras.layers.Dropout(dropout_rate),
                        keras.layers.Dense(embedding_dim),
                    ]
                )
            )
            self.norm1_layers.append(keras.layers.LayerNormalization())
            self.norm2_layers.append(keras.layers.LayerNormalization())
            self.dropout_layers.append(keras.layers.Dropout(dropout_rate))

        self.mu_head = keras.layers.Dense(embedding_dim, name="ib_mu")
        self.logvar_head = keras.layers.Dense(embedding_dim, name="ib_logvar")
        self.output_norm = keras.layers.LayerNormalization(name="ib_output_norm")

    def call(self, modality_embeddings, training=False):
        """modality_embeddings: [batch, num_modalities, embedding_dim]."""
        x = modality_embeddings + self.positional_embedding

        for i in range(self.num_layers):
            attn_out = self.attn_layers[i](x, x, training=training)
            x = self.norm1_layers[i](
                x + self.dropout_layers[i](attn_out, training=training)
            )
            ffn_out = self.ffn_layers[i](x, training=training)
            x = self.norm2_layers[i](
                x + self.dropout_layers[i](ffn_out, training=training)
            )

        pooled = tf.reduce_mean(x, axis=1)
        mu = self.mu_head(pooled)
        logvar = tf.clip_by_value(self.logvar_head(pooled), -8.0, 4.0)

        if training:
            eps = tf.random.normal(tf.shape(mu))
            z = mu + tf.exp(0.5 * logvar) * eps
        else:
            z = mu

        kl = -0.5 * tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=1)
        kl_mean = tf.reduce_mean(kl)
        self.last_kl = kl_mean
        self.add_loss(self.ib_beta * kl_mean)

        return self.output_norm(z)
