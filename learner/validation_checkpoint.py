import math
from typing import Optional

import tensorflow as tf

from data_loaders.pain_ds_config import (
    SUPPORTED_VALIDATION_CHECKPOINT_METRICS,
    VALIDATION_CHECKPOINT_MODES,
)


VALIDATION_CHECKPOINT_MINIMIZE_METRICS = {
    "loss",
    "task_loss",
    "contrastive_loss",
    "triplet_loss",
    "can_local_loss",
    "can_global_loss",
    "can_margin_loss",
    "inter_class_similarity",
}


class ValidationCheckpointTracker:
    """Keep the best validation model/optimizer state for one fold."""

    def __init__(self, metric: str, mode: str = "auto") -> None:
        self.metric = str(metric).strip()
        self.mode = str(mode).strip()
        if self.metric not in SUPPORTED_VALIDATION_CHECKPOINT_METRICS:
            raise ValueError(
                "metric must be one of: "
                + ", ".join(SUPPORTED_VALIDATION_CHECKPOINT_METRICS)
            )
        if self.mode not in VALIDATION_CHECKPOINT_MODES:
            raise ValueError(
                "mode must be one of: " + ", ".join(VALIDATION_CHECKPOINT_MODES)
            )

        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.best_step: Optional[int] = None
        self.best_metrics: dict[str, float] = {}
        self._model_snapshot: Optional[list[tf.Tensor]] = None
        self._optimizer_snapshot: Optional[list[tf.Tensor]] = None

    @property
    def resolved_mode(self) -> str:
        if self.mode == "auto":
            return (
                "min"
                if self.metric in VALIDATION_CHECKPOINT_MINIMIZE_METRICS
                else "max"
            )
        return self.mode

    @property
    def has_checkpoint(self) -> bool:
        return self._model_snapshot is not None

    @staticmethod
    def _snapshot_variables(variables: list[tf.Variable]) -> list[tf.Tensor]:
        return [tf.identity(variable) for variable in variables]

    @staticmethod
    def _restore_snapshot(
        variables: list[tf.Variable],
        snapshot: list[tf.Tensor],
        *,
        label: str,
    ) -> None:
        if len(variables) != len(snapshot):
            raise RuntimeError(
                f"Cannot restore {label}: variable count changed from "
                f"{len(snapshot)} to {len(variables)}."
            )
        for variable, value in zip(variables, snapshot):
            variable.assign(value)

    def is_better(self, value: float) -> bool:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(numeric_value):
            return False
        if self.best_value is None:
            return True
        if self.resolved_mode == "min":
            return numeric_value < self.best_value
        return numeric_value > self.best_value

    def maybe_update(
        self,
        *,
        value: float,
        epoch: int,
        step: int,
        metrics: dict[str, float],
        model_variables: list[tf.Variable],
        optimizer_variables: list[tf.Variable],
    ) -> bool:
        if not self.is_better(value):
            return False

        self.best_value = float(value)
        self.best_epoch = int(epoch)
        self.best_step = int(step)
        self.best_metrics = {str(key): float(item) for key, item in metrics.items()}
        self._model_snapshot = self._snapshot_variables(model_variables)
        self._optimizer_snapshot = self._snapshot_variables(optimizer_variables)
        return True

    def restore(
        self,
        *,
        model_variables: list[tf.Variable],
        optimizer_variables: list[tf.Variable],
    ) -> bool:
        if self._model_snapshot is None or self._optimizer_snapshot is None:
            return False
        self._restore_snapshot(
            model_variables,
            self._model_snapshot,
            label="validation checkpoint model variables",
        )
        self._restore_snapshot(
            optimizer_variables,
            self._optimizer_snapshot,
            label="validation checkpoint optimizer variables",
        )
        return True

    def summary(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "mode": self.mode,
            "resolved_mode": self.resolved_mode,
            "value": self.best_value,
            "epoch": self.best_epoch,
            "step": self.best_step,
            "metrics": dict(self.best_metrics),
        }
