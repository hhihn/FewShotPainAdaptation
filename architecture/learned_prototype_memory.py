import tensorflow as tf
from tensorflow import keras


class LearnedPrototypeMemory(keras.layers.Layer):
    """Store trainable class-conditional temporal prototype slots.

    The layer broadcasts learned prototype feature maps across tasks so CAN can
    compare query maps against memory slots instead of sampled support examples.
    """

    def __init__(
        self,
        num_classes: int,
        slots_per_class: int,
        seed: int = 0,
        name: str = "learned_prototype_memory",
    ):
        """Initialize the learned prototype memory.

        Args:
            num_classes: Number of task classes represented in memory.
            slots_per_class: Number of trainable prototype slots per class.
            seed: Initializer seed for prototype maps.
            name: Keras layer name.
        """
        super().__init__(name=name)
        self.num_classes = int(num_classes)
        self.slots_per_class = int(slots_per_class)
        self.seed = int(seed)
        self.prototype_maps = None

    def build(self, input_shape):
        """Create prototype-map weights from query feature-map shape.

        Args:
            input_shape: Query feature-map shape ending in [time, channels].
        """
        feature_time = int(input_shape[-2])
        feature_dim = int(input_shape[-1])
        self.prototype_maps = self.add_weight(
            name="prototype_maps",
            shape=(
                self.num_classes,
                self.slots_per_class,
                feature_time,
                feature_dim,
            ),
            initializer=keras.initializers.GlorotUniform(seed=self.seed),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, query_feature_maps: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Return task-broadcast prototype maps and their class labels.

        Args:
            query_feature_maps: Query maps with shape [tasks, queries, time,
                channels], used only to infer the task count.
        """
        task_count = tf.shape(query_feature_maps)[0]
        slot_count = self.num_classes * self.slots_per_class
        flat_maps = tf.reshape(
            self.prototype_maps,
            [
                slot_count,
                tf.shape(self.prototype_maps)[2],
                tf.shape(self.prototype_maps)[3],
            ],
        )
        task_maps = tf.broadcast_to(
            flat_maps[tf.newaxis, ...],
            [
                task_count,
                slot_count,
                tf.shape(flat_maps)[1],
                tf.shape(flat_maps)[2],
            ],
        )
        labels = tf.repeat(
            tf.range(self.num_classes, dtype=tf.int32),
            repeats=self.slots_per_class,
        )
        labels = tf.broadcast_to(labels[tf.newaxis, :], [task_count, slot_count])
        return task_maps, labels
