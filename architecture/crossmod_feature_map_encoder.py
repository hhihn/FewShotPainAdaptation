import tensorflow as tf
from tensorflow import keras

from architecture.fourier_positional_embedding import FourierPositionalEncoding1D
from architecture.crossmod_encoder_layer import CrossModTransformerEncoderLayer
from architecture.eegnet_style_encoder import EEGNetStyleEncoder


class CrossModFeatureMapEncoder(keras.Model):
    """Fuse EDA and ECG feature maps with CrossMod attention.

    Per-modality EEGNet-style frontends extract temporal maps before modality
    self-attention and bidirectional cross-attention produce fused maps.
    """

    def __init__(
        self,
        sequence_length: int = 2500,
        num_sensors: int = 2,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        separable_filters: int = 16,
        temporal_kernel_size: int = 64,
        separable_kernel_size: int = 16,
        pool_size_1: int = 4,
        pool_size_2: int = 8,
        dropout_rate: float = 0.25,
        l2_weight: float = 1e-4,
        normalization: str = "group",
        group_norm_groups: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 2,
        positional_base: float = 10000.0,
        attention_dropout_rate: float = 0.0,
        ff_activation: str = "relu",
        name: str = "crossmod_encoder",
    ):
        """Initialize the CrossMod feature-map encoder.

        Args:
            sequence_length: Number of time steps in each input window.
            num_sensors: Number of input sensor channels; must be 2.
            temporal_filters: Temporal filters for each EEGNet frontend.
            depth_multiplier: Depth multiplier for per-sensor frontend mixing.
            separable_filters: Output channel count for each frontend map.
            temporal_kernel_size: Kernel size for the first temporal filter.
            separable_kernel_size: Kernel size for separable temporal filtering.
            pool_size_1: First temporal pooling factor.
            pool_size_2: Second temporal pooling factor.
            dropout_rate: Dropout rate inside each frontend.
            l2_weight: L2 regularization weight passed to each frontend.
            normalization: Subject-invariant frontend normalization type.
            group_norm_groups: Preferred number of frontend GroupNorm groups.
            num_heads: Number of attention heads.
            hidden_dim: Hidden width of modality Transformer layers.
            num_layers: Number of modality Transformer layers.
            positional_base: Base used by Fourier positional encodings.
            attention_dropout_rate: Dropout rate for attention layers.
            ff_activation: Activation used by modality feed-forward layers.
            name: Keras model name.
        """
        super().__init__(name=name)
        self.sequence_length = int(sequence_length)
        self.num_sensors = int(num_sensors)
        self.temporal_filters = int(temporal_filters)
        self.depth_multiplier = int(depth_multiplier)
        self.separable_filters = int(separable_filters)
        self.temporal_kernel_size = int(temporal_kernel_size)
        self.separable_kernel_size = int(separable_kernel_size)
        self.pool_size_1 = int(pool_size_1)
        self.pool_size_2 = int(pool_size_2)
        self.dropout_rate = float(dropout_rate)
        self.l2_weight = float(l2_weight)
        self.normalization = str(normalization)
        self.group_norm_groups = int(group_norm_groups)
        self.num_heads = int(num_heads)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.positional_base = float(positional_base)
        self.attention_dropout_rate = float(attention_dropout_rate)
        self.ff_activation = str(ff_activation)

        self._validate_config()
        self.eda_branch = self._build_frontend_branch("eda")
        self.ecg_branch = self._build_frontend_branch("ecg")
        self.eda_position = FourierPositionalEncoding1D(
            self.separable_filters,
            base=self.positional_base,
            name="crossmod_eda_positional_encoding",
        )
        self.ecg_position = FourierPositionalEncoding1D(
            self.separable_filters,
            base=self.positional_base,
            name="crossmod_ecg_positional_encoding",
        )
        self.eda_transformer_layers = [
            CrossModTransformerEncoderLayer(
                input_dim=self.separable_filters,
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
                input_dim=self.separable_filters,
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                dropout_rate=self.attention_dropout_rate,
                ff_activation=self.ff_activation,
                name=f"crossmod_ecg_transformer_{idx}",
            )
            for idx in range(self.num_layers)
        ]
        self.eda_to_ecg_projection = keras.layers.Dense(
            self.separable_filters,
            name="crossmod_eda_to_ecg_projection",
        )
        self.eda_to_ecg_attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.separable_filters // self.num_heads,
            dropout=self.attention_dropout_rate,
            name="crossmod_eda_to_ecg_attention",
        )
        self.ecg_to_eda_attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.separable_filters // self.num_heads,
            dropout=self.attention_dropout_rate,
            name="crossmod_ecg_to_eda_attention",
        )

    def _validate_config(self) -> None:
        """Validate CrossMod encoder hyperparameters.

        Raises:
            ValueError: If dimensions, dropout rates, or attention divisibility
                constraints are invalid.
        """
        positive_int_fields = {
            "sequence_length": self.sequence_length,
            "num_sensors": self.num_sensors,
            "temporal_filters": self.temporal_filters,
            "depth_multiplier": self.depth_multiplier,
            "separable_filters": self.separable_filters,
            "temporal_kernel_size": self.temporal_kernel_size,
            "separable_kernel_size": self.separable_kernel_size,
            "pool_size_1": self.pool_size_1,
            "pool_size_2": self.pool_size_2,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }
        for name, value in positive_int_fields.items():
            if value <= 0:
                raise ValueError(f"crossmod_{name} must be > 0")
        if self.num_sensors != 2:
            raise ValueError("CrossModFeatureMapEncoder requires num_sensors=2")
        if self.dropout_rate < 0 or self.dropout_rate >= 1:
            raise ValueError("eegnet_dropout_rate must be in [0, 1)")
        if self.attention_dropout_rate < 0 or self.attention_dropout_rate >= 1:
            raise ValueError("crossmod_attention_dropout_rate must be in [0, 1)")
        if self.l2_weight < 0:
            raise ValueError("eegnet_l2_weight must be non-negative")
        if self.positional_base <= 0:
            raise ValueError("crossmod_positional_base must be > 0")
        if self.separable_filters % self.num_heads != 0:
            raise ValueError(
                "eegnet_separable_filters must be divisible by crossmod_num_heads"
            )

    def _build_frontend_branch(self, prefix: str):
        """Build one modality frontend from the shared EEGNet-style encoder.

        Args:
            prefix: Name prefix identifying the modality branch.
        """
        return EEGNetStyleEncoder(
            sequence_length=self.sequence_length,
            num_sensors=1,
            temporal_filters=self.temporal_filters,
            depth_multiplier=self.depth_multiplier,
            separable_filters=self.separable_filters,
            temporal_kernel_size=self.temporal_kernel_size,
            separable_kernel_size=self.separable_kernel_size,
            pool_size_1=self.pool_size_1,
            pool_size_2=self.pool_size_2,
            dropout_rate=self.dropout_rate,
            l2_weight=self.l2_weight,
            normalization=self.normalization,
            group_norm_groups=self.group_norm_groups,
            name=f"{prefix}_eegnet_frontend",
        )

    def extract_modality_feature_maps(self, x, training=False):
        """Extract separate EDA and ECG temporal feature maps.

        Args:
            x: Input tensor with shape [batch, time, 2].
            training: Whether frontend dropout and normalization run in training
                mode.
        """
        if x.shape.rank != 3:
            raise ValueError("CrossMod input must have shape [batch, time, sensors]")
        if x.shape[-1] is not None and int(x.shape[-1]) != 2:
            raise ValueError("CrossMod input must contain exactly EDA and ECG channels")
        eda = x[:, :, 0:1]
        ecg = x[:, :, 1:2]
        eda_features = self.eda_branch.extract_feature_map(eda, training=training)
        ecg_features = self.ecg_branch.extract_feature_map(ecg, training=training)
        tf.debugging.assert_equal(
            tf.shape(eda_features)[1],
            tf.shape(ecg_features)[1],
            message="EDA and ECG frontend temporal lengths must match",
        )
        return eda_features, ecg_features

    def extract_feature_map(self, x, training=False):
        """Return fused CrossMod temporal feature maps.

        EDA and ECG maps are position-encoded, passed through modality-specific
        Transformer layers, and fused with bidirectional cross-attention.
        """
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
        """Return fused feature maps for a batch of EDA/ECG windows.

        Args:
            x: Input tensor with shape [batch, time, 2].
            training: Whether child layers run in training mode.
        """
        return self.extract_feature_map(x, training=training)

    def get_config(self):
        """Return serializable CrossMod encoder configuration.

        The configuration mirrors the constructor arguments used to build the
        frontend, positional encoding, and attention layers.
        """
        return {
            "sequence_length": self.sequence_length,
            "num_sensors": self.num_sensors,
            "temporal_filters": self.temporal_filters,
            "depth_multiplier": self.depth_multiplier,
            "separable_filters": self.separable_filters,
            "temporal_kernel_size": self.temporal_kernel_size,
            "separable_kernel_size": self.separable_kernel_size,
            "pool_size_1": self.pool_size_1,
            "pool_size_2": self.pool_size_2,
            "dropout_rate": self.dropout_rate,
            "l2_weight": self.l2_weight,
            "normalization": self.normalization,
            "group_norm_groups": self.group_norm_groups,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "positional_base": self.positional_base,
            "attention_dropout_rate": self.attention_dropout_rate,
            "ff_activation": self.ff_activation,
        }
