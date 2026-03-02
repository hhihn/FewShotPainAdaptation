import logging
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class _ProgressScope:
    """Common scope fields reused across progress logs."""

    fold_idx: int
    total_folds: int
    epoch_idx: Optional[int] = None
    total_epochs: Optional[int] = None
    step_idx: Optional[int] = None
    total_steps: Optional[int] = None
    step_label: Optional[str] = None


class TrainingProgressReporter:
    """Centralized progress logging for training and evaluation loops."""

    def __init__(
        self,
        logger: logging.Logger,
        train_log_every: int = 10,
        eval_log_every: int = 10,
    ):
        self.logger = logger
        self.train_log_every = max(1, int(train_log_every))
        self.eval_log_every = max(1, int(eval_log_every))

    @staticmethod
    def _pct(current: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return 100.0 * float(current) / float(total)

    @staticmethod
    def _should_log(current: int, total: int, interval: int) -> bool:
        return current == 1 or current % interval == 0 or current == total

    def _compose_scope_prefix(self, scope: _ProgressScope) -> str:
        parts = [
            (
                f"[Fold {scope.fold_idx}/{scope.total_folds} | "
                f"{self._pct(scope.fold_idx, scope.total_folds):.1f}%]"
            )
        ]
        if scope.epoch_idx is not None and scope.total_epochs is not None:
            parts.append(
                f"[Epoch {scope.epoch_idx}/{scope.total_epochs} | "
                f"{self._pct(scope.epoch_idx, scope.total_epochs):.1f}%]"
            )
        if (
            scope.step_idx is not None
            and scope.total_steps is not None
            and scope.step_label is not None
        ):
            parts.append(
                f"[{scope.step_label} {scope.step_idx}/{scope.total_steps} | "
                f"{self._pct(scope.step_idx, scope.total_steps):.1f}%]"
            )
        return " ".join(parts)

    def _log_metric_line(
        self,
        scope: _ProgressScope,
        loss: float,
        metric_value: float,
        metric_name: str,
    ) -> None:
        self.logger.info(
            f"{self._compose_scope_prefix(scope)} "
            f"loss={loss:.4f}, {metric_name}={metric_value:.4f}"
        )

    def log_fold_start(
        self, fold_idx: int, total_folds: int, test_subject: int
    ) -> None:
        scope = _ProgressScope(fold_idx=fold_idx, total_folds=total_folds)
        self.logger.info(
            f"{self._compose_scope_prefix(scope)} Start - held-out subject={test_subject}"
        )

    def log_fold_complete(
        self,
        fold_idx: int,
        total_folds: int,
        test_subject: int,
        test_loss: float,
        test_accuracy: float,
    ) -> None:
        scope = _ProgressScope(fold_idx=fold_idx, total_folds=total_folds)
        self.logger.info(
            f"{self._compose_scope_prefix(scope)} "
            f"Complete - held-out subject={test_subject}, "
            f"test_loss={test_loss:.4f}, test_accuracy={test_accuracy:.4f}"
        )

    def log_step(
        self,
        stage: str,
        fold_idx: int,
        total_folds: int,
        step_idx: int,
        total_steps: int,
        loss: float,
        metric_value: float,
        metric_name: str = "accuracy",
        epoch_idx: Optional[int] = None,
        total_epochs: Optional[int] = None,
        log_every: Optional[int] = None,
    ) -> None:
        interval = self.eval_log_every if log_every is None else max(1, int(log_every))
        if not self._should_log(step_idx, total_steps, interval):
            return
        scope = _ProgressScope(
            fold_idx=fold_idx,
            total_folds=total_folds,
            epoch_idx=epoch_idx,
            total_epochs=total_epochs,
            step_idx=step_idx,
            total_steps=total_steps,
            step_label=stage,
        )
        self._log_metric_line(
            scope=scope,
            loss=loss,
            metric_value=metric_value,
            metric_name=metric_name,
        )

    def log_adaptation_start(
        self,
        fold_idx: int,
        total_folds: int,
        test_subject: int,
        adaptation_steps: int,
    ) -> None:
        scope = _ProgressScope(fold_idx=fold_idx, total_folds=total_folds)
        self.logger.info(
            f"{self._compose_scope_prefix(scope)} "
            f"Adapting on held-out subject={test_subject} for {adaptation_steps} gradient steps"
        )

    def log_subject_summary(
        self,
        stage: str,
        fold_idx: int,
        total_folds: int,
        test_subject: int,
        loss: float,
        metrics: dict,
    ) -> None:
        scope = _ProgressScope(fold_idx=fold_idx, total_folds=total_folds)
        self.logger.info(
            f"{self._compose_scope_prefix(scope)} "
            f"[{stage} subject={test_subject}] "
            f"loss={loss:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}"
        )
