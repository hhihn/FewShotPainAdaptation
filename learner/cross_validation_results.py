import numpy as np

from utils.training_progress_csv import TrainingProgressCSVWriter


class CrossValidationResultRecorder:
    """Own CSV progress writes and LOSO result aggregation.

    The recorder keeps the nested result payload synchronized with progress CSV
    events emitted during training, validation, adaptation, and held-out testing.
    """

    def __init__(
        self,
        *,
        heldout_eval_pairs: list[tuple[int, int]],
        training_progress_output_dir: str,
        csv_flush_every_events: int,
        logger,
        validation_checkpoint_metric: str = "accuracy",
        validation_checkpoint_mode: str = "auto",
    ):
        """Initialize the result recorder.

        Args:
            heldout_eval_pairs: Held-out k/q task sizes to track separately.
            training_progress_output_dir: Directory for per-fold progress CSVs.
            csv_flush_every_events: Flush cadence for the progress writer.
            validation_checkpoint_metric: Metric tracked by validation checkpoints.
            validation_checkpoint_mode: Optimization mode for validation checkpoints.
            logger: Logger used for aggregate summaries.
        """
        self.heldout_eval_pairs = [
            (int(k_shot), int(q_query)) for k_shot, q_query in heldout_eval_pairs
        ]
        self.logger = logger
        self.validation_checkpoint_metric = validation_checkpoint_metric
        self.validation_checkpoint_mode = validation_checkpoint_mode
        self.csv_writer = TrainingProgressCSVWriter(
            output_dir=training_progress_output_dir,
            flush_every_events=max(1, int(csv_flush_every_events)),
        )
        self.current_progress_file = None
        self.cv_results = self._build_initial_results()

    @property
    def results(self) -> dict:
        """Return the mutable cross-validation result dictionary.

        Callers use this property to attach the final payload to reports.
        """
        return self.cv_results

    def _build_initial_results(self) -> dict:
        """Build the initial nested cross-validation result payload.

        The payload contains aggregate fold lists, held-out task-size buckets,
        progress file paths, and architecture metadata.
        """
        return {
            "train_losses": [],
            "train_accuracies": [],
            "val_losses": [],
            "val_accuracies": [],
            "test_losses": [],
            "test_accuracies": [],
            "zero_shot_losses": [],
            "zero_shot_accuracies": [],
            "zero_shot_precisions": [],
            "zero_shot_recalls": [],
            "zero_shot_f1s": [],
            "zero_shot_intra_class_similarities": [],
            "zero_shot_inter_class_similarities": [],
            "zero_shot_can_true_class_scores": [],
            "zero_shot_can_best_other_scores": [],
            "zero_shot_can_score_margins": [],
            "k_shot_losses": [],
            "k_shot_accuracies": [],
            "k_shot_precisions": [],
            "k_shot_recalls": [],
            "k_shot_f1s": [],
            "k_shot_intra_class_similarities": [],
            "k_shot_inter_class_similarities": [],
            "k_shot_can_true_class_scores": [],
            "k_shot_can_best_other_scores": [],
            "k_shot_can_score_margins": [],
            "heldout_eval_task_sizes": [
                {"k_shot": int(k_shot), "q_query": int(q_query)}
                for k_shot, q_query in self.heldout_eval_pairs
            ],
            "heldout_eval_by_task_size": {
                self.size_key(k_shot, q_query): self._build_size_bucket(
                    k_shot,
                    q_query,
                )
                for k_shot, q_query in self.heldout_eval_pairs
            },
            "training_progress_files": [],
            "can_alignment_summary_files": [],
            "can_sample_statistics_files": [],
            "can_feature_export_files": [],
            "model_architecture_file": None,
            "validation_checkpoint_metric": self.validation_checkpoint_metric,
            "validation_checkpoint_mode": self.validation_checkpoint_mode,
            "validation_checkpoint_values": [],
            "validation_checkpoint_epochs": [],
            "validation_checkpoint_steps": [],
            "validation_checkpoint_metrics": [],
        }

    @staticmethod
    def size_key(k_shot: int, q_query: int) -> str:
        """Return the stable key for a held-out task-size bucket.

        Args:
            k_shot: Support samples per class.
            q_query: Query samples per class.
        """
        return f"k{int(k_shot)}_q{int(q_query)}"

    @staticmethod
    def _build_size_bucket(k_shot: int, q_query: int) -> dict:
        """Build an empty result bucket for one held-out task size.

        Args:
            k_shot: Support samples per class.
            q_query: Query samples per class.
        """
        return {
            "k_shot": int(k_shot),
            "q_query": int(q_query),
            "zero_shot_losses": [],
            "zero_shot_accuracies": [],
            "zero_shot_precisions": [],
            "zero_shot_recalls": [],
            "zero_shot_f1s": [],
            "zero_shot_intra_class_similarities": [],
            "zero_shot_inter_class_similarities": [],
            "zero_shot_can_true_class_scores": [],
            "zero_shot_can_best_other_scores": [],
            "zero_shot_can_score_margins": [],
            "k_shot_losses": [],
            "k_shot_accuracies": [],
            "k_shot_precisions": [],
            "k_shot_recalls": [],
            "k_shot_f1s": [],
            "k_shot_intra_class_similarities": [],
            "k_shot_inter_class_similarities": [],
            "k_shot_can_true_class_scores": [],
            "k_shot_can_best_other_scores": [],
            "k_shot_can_score_margins": [],
        }

    @staticmethod
    def _metric_event_kwargs(
        metrics: dict,
        *,
        include_similarity_margin: bool = False,
    ) -> dict:
        """Convert metric dictionaries into CSV event keyword arguments.

        Args:
            metrics: Evaluation metric dictionary.
            include_similarity_margin: Whether to include non-CAN margin output.
        """
        can_mode = "can_score_margin" in metrics
        event_kwargs = {
            "task_loss": metrics.get("task_loss"),
            "contrastive_loss": None if can_mode else metrics.get("contrastive_loss"),
            "triplet_loss": None if can_mode else metrics.get("triplet_loss"),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "intra_class_similarity": None
            if can_mode
            else metrics.get("intra_class_similarity"),
            "inter_class_similarity": None
            if can_mode
            else metrics.get("inter_class_similarity"),
            "can_local_loss": metrics.get("can_local_loss"),
            "can_global_loss": None if can_mode else metrics.get("can_global_loss"),
            "can_margin_loss": metrics.get("can_margin_loss") if can_mode else None,
            "can_true_class_score": metrics.get("can_true_class_score"),
            "can_best_other_score": metrics.get("can_best_other_score"),
            "can_score_margin": metrics.get("can_score_margin"),
        }
        if include_similarity_margin:
            event_kwargs["similarity_margin"] = (
                None if can_mode else metrics.get("similarity_margin")
            )
        return event_kwargs

    @staticmethod
    def _append_eval_diagnostics(bucket: dict, prefix: str, metrics: dict) -> None:
        """Append similarity or CAN diagnostics to a result bucket.

        Args:
            bucket: Result dictionary receiving metric values.
            prefix: Metric prefix such as ``zero_shot`` or ``k_shot``.
            metrics: Evaluation metric dictionary.
        """
        if "can_score_margin" in metrics:
            bucket[f"{prefix}_can_true_class_scores"].append(
                metrics["can_true_class_score"]
            )
            bucket[f"{prefix}_can_best_other_scores"].append(
                metrics["can_best_other_score"]
            )
            bucket[f"{prefix}_can_score_margins"].append(metrics["can_score_margin"])
        else:
            bucket[f"{prefix}_intra_class_similarities"].append(
                metrics["intra_class_similarity"]
            )
            bucket[f"{prefix}_inter_class_similarities"].append(
                metrics["inter_class_similarity"]
            )

    @staticmethod
    def _append_eval_result(
        bucket: dict,
        *,
        prefix: str,
        loss: float,
        metrics: dict,
    ) -> None:
        """Append one evaluation result to a result bucket.

        Args:
            bucket: Result dictionary receiving metric values.
            prefix: Metric prefix such as ``zero_shot`` or ``k_shot``.
            loss: Evaluation loss value.
            metrics: Evaluation metric dictionary.
        """
        bucket[f"{prefix}_losses"].append(loss)
        bucket[f"{prefix}_accuracies"].append(metrics["accuracy"])
        bucket[f"{prefix}_precisions"].append(metrics["precision"])
        bucket[f"{prefix}_recalls"].append(metrics["recall"])
        bucket[f"{prefix}_f1s"].append(metrics["f1"])
        CrossValidationResultRecorder._append_eval_diagnostics(
            bucket,
            prefix,
            metrics,
        )

    def start_fold(self, *, fold_idx: int, test_subject: int) -> str:
        """Start progress recording for one fold.

        Args:
            fold_idx: One-based fold index.
            test_subject: Held-out subject identifier.
        """
        self.current_progress_file = self.csv_writer.start_fold(
            fold_idx=fold_idx,
            test_subject=test_subject,
        )
        return self.current_progress_file

    def close_fold(self) -> None:
        """Close the active fold writer and record its CSV path.

        The current progress file is appended to the result payload when present.
        """
        self.csv_writer.close()
        if self.current_progress_file is not None:
            self.cv_results["training_progress_files"].append(
                self.current_progress_file
            )
        self.current_progress_file = None

    def set_model_architecture_file(self, architecture_path: str) -> None:
        """Record the saved model architecture summary path.

        Args:
            architecture_path: Path to the architecture summary file.
        """
        self.cv_results["model_architecture_file"] = architecture_path

    def write_train_update(
        self,
        *,
        fold_idx: int,
        test_subject: int,
        epoch: int,
        epoch_total: int,
        step: int,
        step_total: int,
        loss: float,
        task_loss: float,
        contrastive_loss: float,
        triplet_loss: float,
        accuracy: float,
        can_local_loss: float | None = None,
        can_global_loss: float | None = None,
        can_margin_loss: float | None = None,
    ) -> None:
        """Write one training progress update event.

        Args:
            fold_idx: One-based fold index.
            test_subject: Held-out subject identifier.
            epoch: Current epoch number.
            epoch_total: Total epochs.
            step: Current train step within the epoch.
            step_total: Total train steps in the epoch.
            loss: Total objective value.
            task_loss: Supervised task loss value.
            contrastive_loss: Contrastive auxiliary loss value.
            triplet_loss: Triplet auxiliary loss value.
            accuracy: Batch accuracy.
            can_local_loss: Optional CAN local loss value.
            can_global_loss: Optional CAN global loss value.
            can_margin_loss: Optional CAN margin loss value.
        """
        self.csv_writer.write_event(
            fold_idx=fold_idx,
            test_subject=test_subject,
            event_type="train_update",
            epoch=epoch,
            epoch_total=epoch_total,
            step=step,
            step_total=step_total,
            loss=loss,
            task_loss=task_loss,
            contrastive_loss=contrastive_loss,
            triplet_loss=triplet_loss,
            can_local_loss=can_local_loss,
            can_global_loss=can_global_loss,
            can_margin_loss=can_margin_loss,
            accuracy=accuracy,
        )

    def write_validation_step(
        self,
        *,
        fold_idx: int,
        test_subject: int,
        epoch: int,
        epoch_total: int,
        step: int,
        step_total: int,
        loss: float,
        metrics: dict,
        checkpoint_metric: str | None = None,
        checkpoint_value: float | None = None,
        checkpoint_is_best: bool | None = None,
    ) -> None:
        """Write one validation progress event.

        Args:
            fold_idx: One-based fold index.
            test_subject: Held-out subject identifier.
            epoch: Current epoch number.
            epoch_total: Total epochs.
            step: Current train step.
            step_total: Total train steps.
            loss: Validation loss.
            metrics: Validation metric dictionary.
            checkpoint_metric: Name of validation checkpoint metric.
            checkpoint_value: Current checkpoint metric value.
            checkpoint_is_best: Whether current checkpoint improved.
        """
        self.csv_writer.write_event(
            fold_idx=fold_idx,
            test_subject=test_subject,
            event_type="validation_step",
            epoch=epoch,
            epoch_total=epoch_total,
            step=step,
            step_total=step_total,
            loss=loss,
            **self._metric_event_kwargs(metrics, include_similarity_margin=True),
            checkpoint_metric=checkpoint_metric,
            checkpoint_value=checkpoint_value,
            checkpoint_is_best=checkpoint_is_best,
        )

    def write_metric_event(
        self,
        *,
        fold_idx: int,
        test_subject: int,
        event_type: str,
        loss: float,
        metrics: dict,
    ) -> None:
        """Write a generic metric event to the progress CSV.

        Args:
            fold_idx: One-based fold index.
            test_subject: Held-out subject identifier.
            event_type: Progress CSV event type.
            loss: Event loss value.
            metrics: Event metric dictionary.
        """
        self.csv_writer.write_event(
            fold_idx=fold_idx,
            test_subject=test_subject,
            event_type=event_type,
            loss=loss,
            **self._metric_event_kwargs(metrics),
        )

    def write_adaptation_event(
        self,
        *,
        fold_idx: int,
        test_subject: int,
        event_type: str,
        adaptation_losses: list[float],
    ) -> float:
        """Write an adaptation event and return mean adaptation loss.

        Args:
            fold_idx: One-based fold index.
            test_subject: Held-out subject identifier.
            event_type: Progress CSV event type.
            adaptation_losses: Per-step adaptation losses.
        """
        mean_loss = float(np.mean(adaptation_losses)) if adaptation_losses else 0.0
        self.csv_writer.write_event(
            fold_idx=fold_idx,
            test_subject=test_subject,
            event_type=event_type,
            loss=mean_loss,
        )
        return mean_loss

    def record_validation_checkpoint(self, checkpoint_summary: dict) -> None:
        """Append one fold's validation checkpoint summary."""
        self.cv_results["validation_checkpoint_values"].append(
            checkpoint_summary["value"]
        )
        self.cv_results["validation_checkpoint_epochs"].append(
            checkpoint_summary["epoch"]
        )
        self.cv_results["validation_checkpoint_steps"].append(
            checkpoint_summary["step"]
        )
        self.cv_results["validation_checkpoint_metrics"].append(
            checkpoint_summary["metrics"]
        )

    def record_can_alignment_summary_file(self, path: str | None) -> None:
        """Record a generated CAN alignment summary path when present."""
        if path is not None:
            self.cv_results["can_alignment_summary_files"].append(path)

    def record_can_sample_statistics_file(self, path: str | None) -> None:
        """Record a generated CAN sample statistics path when present."""
        if path is not None:
            self.cv_results["can_sample_statistics_files"].append(path)

    def record_can_feature_export_file(self, path: str | None) -> None:
        """Record a generated CAN feature export path when present."""
        if path is not None:
            self.cv_results["can_feature_export_files"].append(path)

    def record_heldout_size_result(
        self,
        *,
        size_key: str,
        zero_shot_loss: float,
        zero_shot_metrics: dict,
        adaptation_losses: list[float],
        k_shot_loss: float,
        k_shot_metrics: dict,
        zero_shot_task_batch: list[dict] | None = None,
        k_shot_task_batch: list[dict] | None = None,
    ) -> dict:
        """Record held-out results for one k/q task-size bucket.

        Returns:
            Compact summary dictionary for the evaluated task size.
        """
        size_results = self.cv_results["heldout_eval_by_task_size"][size_key]
        self._append_eval_result(
            size_results,
            prefix="zero_shot",
            loss=zero_shot_loss,
            metrics=zero_shot_metrics,
        )
        self._append_eval_result(
            size_results,
            prefix="k_shot",
            loss=k_shot_loss,
            metrics=k_shot_metrics,
        )
        return {
            "zero_shot_loss": zero_shot_loss,
            "zero_shot_metrics": zero_shot_metrics,
            "zero_shot_task_batch": zero_shot_task_batch or [],
            "adaptation_mean_loss": (
                float(np.mean(adaptation_losses)) if adaptation_losses else 0.0
            ),
            "k_shot_loss": k_shot_loss,
            "k_shot_metrics": k_shot_metrics,
            "k_shot_task_batch": k_shot_task_batch or [],
        }

    def write_legacy_heldout_events(
        self,
        *,
        fold_idx: int,
        test_subject: int,
        zero_shot_loss: float,
        zero_shot_metrics: dict,
        adaptation_mean_loss: float,
        k_shot_loss: float,
        k_shot_metrics: dict,
        run_adaptation: bool,
    ) -> None:
        """Write legacy zero-shot/adaptation/k-shot summary events.

        These events preserve the historical progress CSV shape expected by
        plotting and reporting scripts.
        """
        self.write_metric_event(
            fold_idx=fold_idx,
            test_subject=test_subject,
            event_type="zero_shot_summary",
            loss=zero_shot_loss,
            metrics=zero_shot_metrics,
        )
        if run_adaptation:
            self.csv_writer.write_event(
                fold_idx=fold_idx,
                test_subject=test_subject,
                event_type="adaptation_phase",
                loss=adaptation_mean_loss,
            )
        self.write_metric_event(
            fold_idx=fold_idx,
            test_subject=test_subject,
            event_type="k_shot_summary",
            loss=k_shot_loss,
            metrics=k_shot_metrics,
        )

    def record_fold_result(
        self,
        *,
        fold_idx: int,
        test_subject: int,
        fold_results: dict,
        zero_shot_loss: float,
        zero_shot_metrics: dict,
        k_shot_loss: float,
        k_shot_metrics: dict,
    ) -> None:
        """Append one fold's aggregate results to the CV payload.

        Args:
            fold_idx: One-based fold index.
            test_subject: Held-out subject identifier.
            fold_results: Training and validation histories for the fold.
            zero_shot_loss: Held-out zero-shot loss.
            zero_shot_metrics: Held-out zero-shot metrics.
            k_shot_loss: Held-out post-adaptation loss.
            k_shot_metrics: Held-out post-adaptation metrics.
        """
        self.cv_results["train_losses"].append(np.mean(fold_results["train_losses"]))
        self.cv_results["train_accuracies"].append(
            np.mean(fold_results["train_accuracies"])
        )
        self.cv_results["val_losses"].append(np.mean(fold_results["val_losses"]))
        self.cv_results["val_accuracies"].append(
            np.mean(fold_results["val_accuracies"])
        )
        self.cv_results["test_losses"].append(zero_shot_loss)
        self.cv_results["test_accuracies"].append(zero_shot_metrics["accuracy"])
        self._append_eval_result(
            self.cv_results,
            prefix="zero_shot",
            loss=zero_shot_loss,
            metrics=zero_shot_metrics,
        )
        self._append_eval_result(
            self.cv_results,
            prefix="k_shot",
            loss=k_shot_loss,
            metrics=k_shot_metrics,
        )
        self.csv_writer.write_event(
            fold_idx=fold_idx,
            test_subject=test_subject,
            event_type="fold_summary",
            loss=zero_shot_loss,
            accuracy=zero_shot_metrics["accuracy"],
        )

    def log_composite_summary(
        self,
        *,
        prefix: str,
        train_metrics: dict,
        val_metrics: dict,
        heldout_metrics: dict,
        elapsed_seconds: float,
    ) -> None:
        """Log a compact train/validation/held-out composite summary.

        Args:
            prefix: Log message prefix.
            train_metrics: Train metric dictionary.
            val_metrics: Validation metric dictionary.
            heldout_metrics: Held-out metric dictionary.
            elapsed_seconds: Runtime used in the summary.
        """
        composite_accuracy = float(
            np.mean(
                [
                    train_metrics["accuracy"],
                    val_metrics["accuracy"],
                    heldout_metrics["accuracy"],
                ]
            )
        )
        self.logger.info(
            f"{prefix}: "
            f"composite={composite_accuracy:.4f}, "
            f"train_acc={train_metrics['accuracy']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"heldout_acc={heldout_metrics['accuracy']:.4f}, "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"heldout_loss={heldout_metrics['loss']:.4f}, "
            f"elapsed_seconds={elapsed_seconds:.2f}"
        )

    def log_cross_validation_aggregate(self, *, title: str) -> None:
        """Log aggregate cross-validation summary statistics.

        Args:
            title: Section title for the aggregate log block.
        """
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(title)
        self.logger.info(f"{'=' * 60}")
        self.logger.info(
            f"Average Zero-shot Accuracy: {np.mean(self.cv_results['zero_shot_accuracies']):.4f} "
            f"(±{np.std(self.cv_results['zero_shot_accuracies']):.4f})"
        )
        self.logger.info(
            f"Average K-shot Accuracy: {np.mean(self.cv_results['k_shot_accuracies']):.4f} "
            f"(±{np.std(self.cv_results['k_shot_accuracies']):.4f})"
        )
        self.logger.info(
            f"Average Zero-shot Loss: {np.mean(self.cv_results['zero_shot_losses']):.4f}"
        )
        self.logger.info(
            f"Average K-shot Loss: {np.mean(self.cv_results['k_shot_losses']):.4f}"
        )
        if self.cv_results.get("zero_shot_can_score_margins"):
            self.logger.info(
                "Average Zero-shot CAN Scores: "
                f"true_class={np.mean(self.cv_results['zero_shot_can_true_class_scores']):.4f}, "
                f"best_other={np.mean(self.cv_results['zero_shot_can_best_other_scores']):.4f}, "
                f"margin={np.mean(self.cv_results['zero_shot_can_score_margins']):.4f}"
            )
            self.logger.info(
                "Average K-shot CAN Scores: "
                f"true_class={np.mean(self.cv_results['k_shot_can_true_class_scores']):.4f}, "
                f"best_other={np.mean(self.cv_results['k_shot_can_best_other_scores']):.4f}, "
                f"margin={np.mean(self.cv_results['k_shot_can_score_margins']):.4f}"
            )
        elif self.cv_results.get("zero_shot_intra_class_similarities"):
            self.logger.info(
                "Average Zero-shot Similarities: "
                f"intra_class={np.mean(self.cv_results['zero_shot_intra_class_similarities']):.4f}, "
                f"inter_class={np.mean(self.cv_results['zero_shot_inter_class_similarities']):.4f}"
            )
            self.logger.info(
                "Average K-shot Similarities: "
                f"intra_class={np.mean(self.cv_results['k_shot_intra_class_similarities']):.4f}, "
                f"inter_class={np.mean(self.cv_results['k_shot_inter_class_similarities']):.4f}"
            )
        self.logger.info(f"{'=' * 60}\n")
