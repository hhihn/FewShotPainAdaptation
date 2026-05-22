import math
import tensorflow as tf
from tensorflow import keras


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
