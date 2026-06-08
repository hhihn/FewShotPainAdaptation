import tensorflow as tf
from tensorflow import keras

from utils.logger import setup_logger


class EEGNetStyleEncoder(keras.Model):
    """Encode physiological windows with a compact EEGNet-style frontend.

    The encoder returns temporal feature maps for attention-based models.
    """

    def __init__(
        self,
        sequence_length: int = 2500,
        num_sensors: int = 3,
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
        """Initialize the EEGNet-style encoder.

        Args:
            sequence_length: Number of time steps in each input window.
            num_sensors: Number of sensor channels in each window.
            temporal_filters: Number of first-stage temporal filters.
            depth_multiplier: Depth multiplier for sensor-wise depthwise mixing.
            separable_filters: Number of separable temporal filters.
            temporal_kernel_size: Kernel size for first-stage temporal filters.
            separable_kernel_size: Kernel size for separable temporal filters.
            pool_size_1: Pooling factor after depthwise sensor mixing.
            pool_size_2: Pooling factor after separable temporal filtering.
            dropout_rate: Dropout rate applied after each pooling stage.
            l2_weight: Reserved regularization weight for frontend compatibility.
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
        self.temporal_norm = keras.layers.BatchNormalization(name="temporal_norm")
        self.temporal_activation = keras.layers.Activation("gelu", name="temporal_elu")
        self.depthwise_conv = keras.layers.DepthwiseConv2D(
            kernel_size=(1, self.num_sensors),
            depth_multiplier=self.depth_multiplier,
            use_bias=False,
            depthwise_constraint=keras.constraints.max_norm(1.0),
            name="sensor_depthwise_conv",
        )
        self.depthwise_norm = keras.layers.BatchNormalization(name="depthwise_norm")
        self.depthwise_activation = keras.layers.Activation(
            "gelu", name="depthwise_elu"
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
        self.separable_norm = keras.layers.BatchNormalization(name="separable_norm")
        self.separable_activation = keras.layers.Activation(
            "gelu", name="separable_elu"
        )
        self.pool_2 = keras.layers.AveragePooling2D(
            pool_size=(self.pool_size_2, 1),
            name="separable_average_pool",
        )
        self.dropout_2 = keras.layers.Dropout(
            rate=self.dropout_rate, name="separable_dropout"
        )
        self.logger.debug(
            "Initialized EEGNetStyleEncoder with temporal_filters=%s, "
            "depth_multiplier=%s, separable_filters=%s",
            self.temporal_filters,
            self.depth_multiplier,
            self.separable_filters,
        )

    def _feature_map_4d(self, x, training=False):
        """Return post-separable EEGNet activations.

        Args:
            x: Input tensor with shape [batch, time, sensors].
            training: Whether dropout and normalization run in training mode.

        Returns:
            Tensor with shape [batch, time, 1, channels].
        """
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
        return x

    def extract_feature_map(self, x, training=False):
        """Return post-sensor-mixing temporal features.

        Args:
            x: Input tensor with shape [batch, time, sensors].
            training: Whether child layers run in training mode.
        """
        return self._feature_map_4d(x, training=training)[:, :, 0, :]

    def call(self, x, training=False):
        """Encode windows into temporal feature maps.

        Args:
            x: Input tensor with shape [batch, time, sensors].
            training: Whether child layers run in training mode.
        """
        return self.extract_feature_map(x, training=training)

    def get_config(self):
        """Return model configuration for serialization.

        The returned dictionary contains constructor-compatible hyperparameters
        for rebuilding this encoder.
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
        }
