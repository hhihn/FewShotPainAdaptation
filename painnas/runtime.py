"""TensorFlow lifecycle helpers for independent trials and LOSO folds."""

from __future__ import annotations

import gc
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras


def reset_runtime(seed: int, *, deterministic: bool = True) -> None:
    """Drop Keras state and seed a fresh model/optimizer lifecycle."""

    keras.backend.clear_session()
    gc.collect()
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def require_gpu(*, allow_cpu: bool) -> list[str]:
    devices = tf.config.list_physical_devices("GPU")
    names = [device.name for device in devices]
    if not names and not allow_cpu:
        raise RuntimeError(
            "No TensorFlow GPU is available. Use a GPU runtime or pass --allow-cpu "
            "for smoke/debug runs."
        )
    return names
