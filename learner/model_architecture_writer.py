import io
import os

import tensorflow as tf


class ModelArchitectureWriter:
    """Write model and encoder architecture summaries."""

    def __init__(self, *, model_getter):
        self._model_getter = model_getter

    @property
    def model(self):
        return self._model_getter()

    def save_model_architecture(self, sample_task: dict, output_path: str) -> str:
        """Build model and save model architecture summaries to a text file."""
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
