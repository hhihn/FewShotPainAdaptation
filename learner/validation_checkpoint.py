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
    """Track the best validation checkpoint for one training fold.

    The tracker snapshots model and optimizer variables whenever the configured
    validation metric improves.
    """

    def __init__(self, metric: str, mode: str = "auto") -> None:
        """Initialize checkpoint tracking for one validation metric.

        Args:
            metric: Metric name to monitor.
            mode: ``auto``, ``min``, or ``max`` improvement direction.
        """
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
        """Return the concrete checkpoint direction.

        Auto mode minimizes known loss-like metrics and maximizes all others.
        """
        if self.mode == "auto":
            return (
                "min"
                if self.metric in VALIDATION_CHECKPOINT_MINIMIZE_METRICS
                else "max"
            )
        return self.mode

    @property
    def has_checkpoint(self) -> bool:
        """Return whether a model snapshot has been captured.

        A true value means ``restore`` can attempt to assign saved variables.
        """
        return self._model_snapshot is not None

    @staticmethod
    def _snapshot_variables(variables: list[tf.Variable]) -> list[tf.Tensor]:
        """Copy variable values into immutable tensors.

        Args:
            variables: TensorFlow variables to snapshot.
        """
        return [tf.identity(variable) for variable in variables]

    @staticmethod
    def _restore_snapshot(
        variables: list[tf.Variable],
        snapshot: list[tf.Tensor],
        *,
        label: str,
    ) -> None:
        """Restore variables from a captured tensor snapshot.

        Args:
            variables: Variables to assign.
            snapshot: Tensor values captured earlier.
            label: Human-readable label used in error messages.
        """
        if len(variables) != len(snapshot):
            raise RuntimeError(
                f"Cannot restore {label}: variable count changed from "
                f"{len(snapshot)} to {len(variables)}."
            )
        for variable, value in zip(variables, snapshot):
            variable.assign(value)

    def is_better(self, value: float) -> bool:
        """Return whether a metric value improves on the current best.

        Non-finite or non-numeric values are never considered improvements.
        """
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
        """Snapshot model state if the provided metric value improves.

        Returns:
            True when a new checkpoint was captured, otherwise False.
        """
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
        """Restore the best captured model and optimizer state.

        Returns:
            True when snapshots existed and were restored.
        """
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
        """Return serializable metadata for the best checkpoint.

        The summary omits variable tensors and includes only metric metadata.
        """
        return {
            "metric": self.metric,
            "mode": self.mode,
            "resolved_mode": self.resolved_mode,
            "value": self.best_value,
            "epoch": self.best_epoch,
            "step": self.best_step,
            "metrics": dict(self.best_metrics),
        }
