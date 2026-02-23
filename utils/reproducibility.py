import os
import random
import logging

import numpy as np
import tensorflow as tf


def set_global_reproducibility(
    seed: int, deterministic_ops: bool = True, logger: logging.Logger | None = None
) -> None:
    """Set random seeds and deterministic flags for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if deterministic_ops:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as exc:  # pragma: no cover
            if logger is not None:
                logger.warning(f"Could not enable TensorFlow op determinism: {exc}")
