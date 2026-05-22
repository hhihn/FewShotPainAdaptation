import tensorflow as tf
from tensorflow import keras

from architecture.fourier_positional_embedding import FourierPositionalEncoding1D
from architecture.crossmod_encoder_layer import CrossModTransformerEncoderLayer

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

        self.eda_branch = self._build_frontend_branch("eda", regularizer=regularizer)
        self.ecg_branch = self._build_frontend_branch("ecg", regularizer=regularizer)
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
                keras.layers.BatchNormalization(
                    name=f"{prefix}_frontend_temporal_norm"
                ),
                keras.layers.Activation(
                    "gelu", name=f"{prefix}_frontend_temporal_gelu"
                ),
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
                keras.layers.BatchNormalization(
                    name=f"{prefix}_frontend_separable_norm"
                ),
                keras.layers.Activation(
                    "gelu", name=f"{prefix}_frontend_separable_gelu"
                ),
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
