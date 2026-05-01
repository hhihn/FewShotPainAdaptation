import os
import random
import logging

import numpy as np
import tensorflow as tf


def set_global_reproducibility(
    seed: int, deterministic_ops: bool = True, logger: logging.Logger | None = None
) -> None:
    """Set random seeds and deterministic flags for reproducible runs."""
    previous_hash_seed = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    if deterministic_ops:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as exc:  # pragma: no cover
            if logger is not None:
                logger.warning(f"Could not enable TensorFlow op determinism: {exc}")

    if (
        previous_hash_seed not in {None, str(seed)}
        and logger is not None
    ):
        logger.warning(
            "PYTHONHASHSEED changed after interpreter startup; "
            "restart the Python process with the desired value for full hash determinism."
        )
