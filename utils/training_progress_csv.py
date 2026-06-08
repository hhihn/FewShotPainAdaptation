import csv
import os
import time
from typing import Optional


class TrainingProgressCSVWriter:
    """Write per-step training/evaluation progress for each fold to CSV."""

    def __init__(
        self,
        output_dir: str = "outputs/training_progress",
        flush_every_events: int = 100,
    ):
        self.output_dir = output_dir
        self.flush_every_events = max(1, int(flush_every_events))
        self._file = None
        self._writer = None
        self._start_time = 0.0
        self._time_index = 0
        self.current_path: Optional[str] = None

    def start_fold(self, fold_idx: int, test_subject: int) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        self.current_path = os.path.join(
            self.output_dir,
            f"fold_{fold_idx:02d}_subject_{int(test_subject):02d}_training_progress.csv",
        )
        self._file = open(self.current_path, "w", newline="", encoding="utf-8")
        fieldnames = [
            "time_index",
            "elapsed_seconds",
            "fold",
            "test_subject",
            "event_type",
            "epoch",
            "epoch_total",
            "step",
            "step_total",
            "loss",
            "task_loss",
            "can_local_loss",
            "can_margin_loss",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "can_true_class_score",
            "can_best_other_score",
            "can_score_margin",
            "checkpoint_metric",
            "checkpoint_value",
            "checkpoint_is_best",
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._start_time = time.perf_counter()
        self._time_index = 0
        return self.current_path

    def write_event(
        self,
        fold_idx: int,
        test_subject: int,
        event_type: str,
        loss: float,
        task_loss: Optional[float] = None,
        can_local_loss: Optional[float] = None,
        can_margin_loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1: Optional[float] = None,
        can_true_class_score: Optional[float] = None,
        can_best_other_score: Optional[float] = None,
        can_score_margin: Optional[float] = None,
        checkpoint_metric: Optional[str] = None,
        checkpoint_value: Optional[float] = None,
        checkpoint_is_best: Optional[bool] = None,
        epoch: Optional[int] = None,
        epoch_total: Optional[int] = None,
        step: Optional[int] = None,
        step_total: Optional[int] = None,
    ) -> None:
        if self._writer is None:
            return
        self._time_index += 1
        self._writer.writerow(
            {
                "time_index": self._time_index,
                "elapsed_seconds": time.perf_counter() - self._start_time,
                "fold": fold_idx,
                "test_subject": int(test_subject),
                "event_type": event_type,
                "epoch": epoch,
                "epoch_total": epoch_total,
                "step": step,
                "step_total": step_total,
                "loss": loss,
                "task_loss": task_loss,
                "can_local_loss": can_local_loss,
                "can_margin_loss": can_margin_loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "can_true_class_score": can_true_class_score,
                "can_best_other_score": can_best_other_score,
                "can_score_margin": can_score_margin,
                "checkpoint_metric": checkpoint_metric,
                "checkpoint_value": checkpoint_value,
                "checkpoint_is_best": checkpoint_is_best,
            }
        )
        if self._time_index % self.flush_every_events == 0 and self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
        self._file = None
        self._writer = None
