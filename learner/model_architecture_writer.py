import io
import os

import tensorflow as tf


class ModelArchitectureWriter:
    """Write model and encoder architecture summaries.

    The writer builds the model from a sample task before persisting Keras
    summaries to a text file.
    """

    def __init__(self, *, model_getter):
        """Initialize the architecture writer.

        Args:
            model_getter: Callable returning the active Keras model.
        """
        self._model_getter = model_getter

    @property
    def model(self):
        """Return the current model from the injected getter.

        The property avoids storing stale model references across folds.
        """
        return self._model_getter()

    def save_model_architecture(self, sample_task: dict, output_path: str) -> str:
        """Build the model and write architecture summaries.

        Args:
            sample_task: Task dictionary used to build model variables.
            output_path: Text file path for the saved summaries.
        """
        support_x = tf.constant(sample_task["support_X"], dtype=tf.float32)
        support_y = tf.constant(sample_task["support_y"], dtype=tf.int32)
        query_x = tf.constant(sample_task["query_X"], dtype=tf.float32)
        _ = self.model(support_x, support_y, query_x, training=False)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fp:
            fp.write("=== MultimodalPrototypicalNetwork Summary ===\n")
            full_summary = io.StringIO()
            self.model.summary(print_fn=lambda line: full_summary.write(line + "\n"))
            fp.write(full_summary.getvalue())
            fp.write("\n")

            fp.write("=== EEGNet Encoder Summary ===\n")
            encoder_summary = io.StringIO()
            self.model.encoder.summary(
                print_fn=lambda line: encoder_summary.write(line + "\n")
            )
            fp.write(encoder_summary.getvalue())
        print(self.model.summary())
        return output_path
