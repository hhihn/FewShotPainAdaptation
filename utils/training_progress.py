import logging


class TrainingProgressReporter:
    """Centralized progress logging for training and evaluation loops."""

    def __init__(
        self,
        logger: logging.Logger,
        train_log_every: int = 10,
        eval_log_every: int = 5,
        adaptation_log_every: int = 1,
    ):
        self.logger = logger
        self.train_log_every = max(1, int(train_log_every))
        self.eval_log_every = max(1, int(eval_log_every))
        self.adaptation_log_every = max(1, int(adaptation_log_every))

    @staticmethod
    def _pct(current: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return 100.0 * float(current) / float(total)

    @staticmethod
    def _should_log(current: int, total: int, interval: int) -> bool:
        return current == 1 or current % interval == 0 or current == total

    def log_fold_start(
        self, fold_idx: int, total_folds: int, test_subject: int
    ) -> None:
        fold_pct = self._pct(fold_idx, total_folds)
        self.logger.info(
            f"[Fold {fold_idx}/{total_folds} | {fold_pct:.1f}%] Start - held-out subject={test_subject}"
        )

    def log_fold_complete(
        self,
        fold_idx: int,
        total_folds: int,
        test_subject: int,
        test_loss: float,
        test_accuracy: float,
    ) -> None:
        fold_pct = self._pct(fold_idx, total_folds)
        self.logger.info(
            f"[Fold {fold_idx}/{total_folds} | {fold_pct:.1f}%] "
            f"Complete - held-out subject={test_subject}, "
            f"test_loss={test_loss:.4f}, test_accuracy={test_accuracy:.4f}"
        )

    def log_train_step(
        self,
        fold_idx: int,
        total_folds: int,
        epoch_idx: int,
        total_epochs: int,
        episode_idx: int,
        total_episodes: int,
        loss: float,
        metric_value: float,
        metric_name: str = "accuracy",
    ) -> None:
        if not self._should_log(episode_idx, total_episodes, self.train_log_every):
            return
        fold_pct = self._pct(fold_idx, total_folds)
        epoch_pct = self._pct(epoch_idx, total_epochs)
        episode_pct = self._pct(episode_idx, total_episodes)
        self.logger.info(
            f"[Fold {fold_idx}/{total_folds} | {fold_pct:.1f}%] "
            f"[Epoch {epoch_idx}/{total_epochs} | {epoch_pct:.1f}%] "
            f"[Train episode {episode_idx}/{total_episodes} | {episode_pct:.1f}%] "
            f"loss={loss:.4f}, {metric_name}={metric_value:.4f}"
        )

    def log_eval_step(
        self,
        stage: str,
        fold_idx: int,
        total_folds: int,
        step_idx: int,
        total_steps: int,
        loss: float,
        metric_value: float,
        metric_name: str = "accuracy",
    ) -> None:
        if not self._should_log(step_idx, total_steps, self.eval_log_every):
            return
        fold_pct = self._pct(fold_idx, total_folds)
        step_pct = self._pct(step_idx, total_steps)
        self.logger.info(
            f"[Fold {fold_idx}/{total_folds} | {fold_pct:.1f}%] "
            f"[{stage} {step_idx}/{total_steps} | {step_pct:.1f}%] "
            f"loss={loss:.4f}, {metric_name}={metric_value:.4f}"
        )

    def log_adaptation_step(
        self,
        fold_idx: int,
        total_folds: int,
        test_subject: int,
        step_idx: int,
        total_steps: int,
        loss: float,
        metrics: dict,
    ) -> None:
        if not self._should_log(step_idx, total_steps, self.adaptation_log_every):
            return
        fold_pct = self._pct(fold_idx, total_folds)
        step_pct = self._pct(step_idx, total_steps)
        metrics_str = (
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}"
        )
        self.logger.info(
            f"[Fold {fold_idx}/{total_folds} | {fold_pct:.1f}%] "
            f"[Subject {test_subject} adaptation step {step_idx}/{total_steps} | {step_pct:.1f}%] "
            f"loss={loss:.4f}, {metrics_str}"
        )
