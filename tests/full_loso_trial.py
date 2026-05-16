import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

# Allow running this file directly from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loaders.pain_ds_config import (
    PainDatasetConfig,
    SUPPORTED_VALIDATION_CHECKPOINT_METRICS,
    VALIDATION_CHECKPOINT_MODES,
)
from learner.few_shot_pain_learner import FewShotPainLearner
from utils.logger import setup_logger


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def _to_jsonable(value: Any) -> Any:
    """Convert numpy-heavy payload values to plain JSON-serializable types."""
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metric_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0))}


def _build_summary(cv_results: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "num_folds": int(len(cv_results.get("zero_shot_accuracies", []))),
        "zero_shot_accuracy": _metric_summary(
            cv_results.get("zero_shot_accuracies", [])
        ),
        "k_shot_accuracy": _metric_summary(cv_results.get("k_shot_accuracies", [])),
        "zero_shot_loss": _metric_summary(cv_results.get("zero_shot_losses", [])),
        "k_shot_loss": _metric_summary(cv_results.get("k_shot_losses", [])),
        "zero_shot_f1": _metric_summary(cv_results.get("zero_shot_f1s", [])),
        "k_shot_f1": _metric_summary(cv_results.get("k_shot_f1s", [])),
    }
    transductive_summary_keys = {
        "zero_shot_transductive_losses": "zero_shot_transductive_loss",
        "zero_shot_transductive_accuracies": "zero_shot_transductive_accuracy",
        "zero_shot_transductive_precisions": "zero_shot_transductive_precision",
        "zero_shot_transductive_recalls": "zero_shot_transductive_recall",
        "zero_shot_transductive_f1s": "zero_shot_transductive_f1",
        "k_shot_transductive_losses": "k_shot_transductive_loss",
        "k_shot_transductive_accuracies": "k_shot_transductive_accuracy",
        "k_shot_transductive_precisions": "k_shot_transductive_precision",
        "k_shot_transductive_recalls": "k_shot_transductive_recall",
        "k_shot_transductive_f1s": "k_shot_transductive_f1",
    }
    for key, summary_key in transductive_summary_keys.items():
        values = cv_results.get(key, [])
        if len(values) > 0:
            summary[summary_key] = _metric_summary(values)
    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_full_loso_trial(args: argparse.Namespace) -> dict[str, Any]:
    logger = setup_logger("full_loso_trial")
    if args.logging_verbosity <= 0:
        logger.setLevel(30)
    elif args.logging_verbosity == 1:
        logger.setLevel(20)
    else:
        logger.setLevel(10)

    start_time = time.perf_counter()
    task_class_ids = _parse_int_tuple(args.task_class_ids)

    logger.info("Stage 1/5: Building run configuration")
    config = PainDatasetConfig(
        seed=args.seed,
        deterministic_ops=bool(args.deterministic_ops),
        dataset_source=args.dataset_source,
        data_variant=args.data_variant,
        task_class_ids=task_class_ids,
        k_shot=args.k_shot,
        q_query=args.q_query,
        task_normalize_mode=args.normalize_mode,
        task_construction_mode=args.task_construction_mode,
        classifier_mode=args.classifier_mode,
        attention_mode=str(getattr(args, "attention_mode", "none")),
        can_attention_temperature=float(
            getattr(args, "can_attention_temperature", 1.0)
        ),
        can_meta_hidden_dim=int(getattr(args, "can_meta_hidden_dim", 32)),
        can_local_loss_weight=float(getattr(args, "can_local_loss_weight", 1.0)),
        can_global_loss_weight=float(getattr(args, "can_global_loss_weight", 0.1)),
        can_transductive_iterations=int(
            getattr(args, "can_transductive_iterations", 3)
        ),
        can_transductive_top_k_per_class=int(
            getattr(args, "can_transductive_top_k_per_class", 1)
        ),
        can_transductive_min_confidence=float(
            getattr(args, "can_transductive_min_confidence", 0.0)
        ),
        train_batch_size=args.task_batch_size,
        embedding_batch_size=max(1, int(getattr(args, "embedding_batch_size", 1))),
        tasks_per_epoch=max(1, int(args.tasks_per_epoch)),
        val_tasks=max(1, int(args.val_tasks)),
        heldout_eval_tasks=max(1, int(args.heldout_eval_tasks)),
        num_epochs=max(1, int(args.num_epochs)),
        k_shot_adaptation_steps=max(0, int(args.k_shot_adaptation_steps)),
        train_log_every=max(1, int(args.train_log_every)),
        eval_log_every=max(1, int(args.eval_log_every)),
        val_batch_size=max(1, int(args.val_batch_size)),
        val_every_n_train_steps=max(1, int(args.val_every_n_train_steps)),
        validation_checkpoint_metric=str(
            getattr(args, "validation_checkpoint_metric", "accuracy")
        ),
        validation_checkpoint_mode=str(
            getattr(args, "validation_checkpoint_mode", "auto")
        ),
        train_prefetch_batches=max(1, int(getattr(args, "train_prefetch_batches", 2))),
        train_progress_write_every_n_batches=max(
            1, int(args.train_progress_write_every_n_batches)
        ),
        csv_flush_every_events=max(1, int(args.csv_flush_every_events)),
        single_loso_fold=False,  # Full LOSO over all available subjects.
        loso_start_index=args.loso_start_index,
        loso_stop_index=args.loso_stop_index,
        embedding_dim=args.embedding_dim,
        eegnet_temporal_filters=args.eegnet_temporal_filters,
        eegnet_depth_multiplier=args.eegnet_depth_multiplier,
        eegnet_separable_filters=args.eegnet_separable_filters,
        eegnet_temporal_kernel_size=args.eegnet_temporal_kernel_size,
        eegnet_separable_kernel_size=args.eegnet_separable_kernel_size,
        eegnet_pool_size_1=args.eegnet_pool_size_1,
        eegnet_pool_size_2=args.eegnet_pool_size_2,
        eegnet_dropout_rate=args.eegnet_dropout_rate,
        eegnet_l2_weight=args.eegnet_l2_weight,
        triplet_loss_weight=float(getattr(args, "triplet_loss_weight", 1.0)),
        triplet_margin=float(getattr(args, "triplet_margin", 0.2)),
        triplet_mining_strategy=str(
            getattr(args, "triplet_mining_strategy", "batch_hard")
        ),
        triplet_center_gradient_clip_norm=float(
            getattr(args, "triplet_center_gradient_clip_norm", 0.01)
        ),
        enable_window_shift_augmentation=not args.disable_window_shift,
        gaussian_noise_std=args.gaussian_noise_std,
        logging_verbosity=args.logging_verbosity,
    )

    logger.info("Stage 2/5: Initializing learner, dataset, and cross-validator")
    learner = FewShotPainLearner(
        config=config,
        data_dir=args.data_dir,
        learning_rate=args.learning_rate,
    )

    if (
        args.max_folds is not None
        and args.max_folds > 0
        and args.loso_start_index is None
        and args.loso_stop_index is None
    ):
        max_folds = int(args.max_folds)
        original_fold_count = len(learner.cv.subjects)
        learner.cv.subjects = list(learner.cv.subjects[:max_folds])
        logger.info(
            "Stage 2.1/5: Limiting fold count for this run "
            f"(requested={max_folds}, original={original_fold_count}, used={len(learner.cv.subjects)})"
        )
    elif args.max_folds is not None and args.max_folds > 0:
        logger.info(
            "Stage 2.1/5: Ignoring --max-folds because an explicit LOSO index range is configured"
        )

    logger.info(
        "Stage 3/5: Starting full LOSO training "
        f"over {len(learner.cv.subjects)} held-out subjects"
    )
    cv_results = learner.train(
        training_progress_output_dir=args.training_progress_output_dir,
        save_model_architecture_first_run=not args.skip_model_architecture_save,
        model_architecture_output_path=args.model_architecture_output,
    )

    logger.info("Stage 4/5: Aggregating fold metrics")
    summary = _build_summary(cv_results)

    payload: dict[str, Any] = {
        "script": "tests/full_loso_trial.py",
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "data_dir": args.data_dir,
        "config": {
            "seed": int(config.seed),
            "dataset_source": str(config.dataset_source),
            "data_variant": str(config.data_variant),
            "task_class_ids": list(config.task_class_ids),
            "k_shot": int(config.k_shot),
            "q_query": int(config.q_query),
            "task_construction_mode": str(config.task_construction_mode),
            "normalize_mode": str(config.task_normalize_mode),
            "classifier_mode": str(config.classifier_mode),
            "attention_mode": str(config.attention_mode),
            "can_attention_temperature": float(config.can_attention_temperature),
            "can_meta_hidden_dim": int(config.can_meta_hidden_dim),
            "can_local_loss_weight": float(config.can_local_loss_weight),
            "can_global_loss_weight": float(config.can_global_loss_weight),
            "can_transductive_iterations": int(config.can_transductive_iterations),
            "can_transductive_top_k_per_class": int(
                config.can_transductive_top_k_per_class
            ),
            "can_transductive_min_confidence": float(
                config.can_transductive_min_confidence
            ),
            "learning_rate": float(args.learning_rate),
            "embedding_dim": int(config.embedding_dim),
            "encoder": "eegnet",
            "eegnet_temporal_filters": int(config.eegnet_temporal_filters),
            "eegnet_depth_multiplier": int(config.eegnet_depth_multiplier),
            "eegnet_separable_filters": int(config.eegnet_separable_filters),
            "eegnet_temporal_kernel_size": int(config.eegnet_temporal_kernel_size),
            "eegnet_separable_kernel_size": int(config.eegnet_separable_kernel_size),
            "eegnet_pool_size_1": int(config.eegnet_pool_size_1),
            "eegnet_pool_size_2": int(config.eegnet_pool_size_2),
            "eegnet_dropout_rate": float(config.eegnet_dropout_rate),
            "eegnet_l2_weight": float(config.eegnet_l2_weight),
            "num_epochs": int(config.num_epochs),
            "tasks_per_epoch": int(config.tasks_per_epoch),
            "train_batch_size": int(config.train_batch_size),
            "embedding_batch_size": int(config.embedding_batch_size),
            "val_tasks": int(config.val_tasks),
            "heldout_eval_tasks": int(config.heldout_eval_tasks),
            "validation_checkpoint_metric": str(config.validation_checkpoint_metric),
            "validation_checkpoint_mode": str(config.validation_checkpoint_mode),
            "k_shot_adaptation_steps": int(config.k_shot_adaptation_steps),
            "window_shift_enabled": bool(config.enable_window_shift_augmentation),
            "gaussian_noise_std": float(config.gaussian_noise_std),
            "triplet_loss_weight": float(config.triplet_loss_weight),
            "triplet_margin": float(config.triplet_margin),
            "triplet_mining_strategy": str(config.triplet_mining_strategy),
            "triplet_center_gradient_clip_norm": float(
                config.triplet_center_gradient_clip_norm
            ),
            "deterministic_ops": bool(config.deterministic_ops),
            "train_progress_write_every_n_batches": int(
                config.train_progress_write_every_n_batches
            ),
            "csv_flush_every_events": int(config.csv_flush_every_events),
            "train_prefetch_batches": int(config.train_prefetch_batches),
            "loso_start_index": config.loso_start_index,
            "loso_stop_index": config.loso_stop_index,
            "max_folds": int(args.max_folds) if args.max_folds is not None else None,
        },
        "summary": summary,
        "cv_results": cv_results,
    }
    payload = _to_jsonable(payload)

    logger.info(f"Stage 5/5: Writing JSON results to {args.output_json}")
    _write_json(Path(args.output_json), payload)
    logger.info(
        "Full LOSO trial complete: "
        f"folds={summary['num_folds']}, "
        f"zero_shot_acc_mean={summary['zero_shot_accuracy']['mean']:.4f}, "
        f"k_shot_acc_mean={summary['k_shot_accuracy']['mean']:.4f}, "
        f"elapsed_seconds={payload['elapsed_seconds']:.2f}"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run full LOSO training/evaluation across all held-out subjects and "
            "store results as JSON."
        )
    )
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument(
        "--dataset-source",
        type=str,
        default="biovid_part_a",
        choices=("painmonit", "biovid_part_a"),
    )
    parser.add_argument(
        "--data-variant",
        type=str,
        default="real",
        choices=("real", "mock"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-shot", type=int, default=10)
    parser.add_argument("--q-query", type=int, default=10)
    parser.add_argument("--task-class-ids", type=str, default="0,4")
    parser.add_argument(
        "--task-construction-mode",
        type=str,
        default="single_subject",
        choices=("single_subject", "cross_subject", "mixed"),
    )
    parser.add_argument(
        "--classifier-mode",
        type=str,
        default="prototype",
        choices=("prototype", "soft_knn"),
    )
    parser.add_argument(
        "--attention-mode",
        type=str,
        default="none",
        choices=("none", "can"),
        help="Optional episodic attention module. 'can' enables CAN/CAM.",
    )
    parser.add_argument("--can-attention-temperature", type=float, default=1.0)
    parser.add_argument("--can-meta-hidden-dim", type=int, default=32)
    parser.add_argument("--can-local-loss-weight", type=float, default=1.0)
    parser.add_argument("--can-global-loss-weight", type=float, default=0.1)
    parser.add_argument("--can-transductive-iterations", type=int, default=3)
    parser.add_argument("--can-transductive-top-k-per-class", type=int, default=1)
    parser.add_argument("--can-transductive-min-confidence", type=float, default=0.0)
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="support",
        choices=("subject", "split", "support", "none"),
    )
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--eegnet-temporal-filters", type=int, default=8)
    parser.add_argument("--eegnet-depth-multiplier", type=int, default=2)
    parser.add_argument("--eegnet-separable-filters", type=int, default=16)
    parser.add_argument("--eegnet-temporal-kernel-size", type=int, default=64)
    parser.add_argument("--eegnet-separable-kernel-size", type=int, default=16)
    parser.add_argument("--eegnet-pool-size-1", type=int, default=4)
    parser.add_argument("--eegnet-pool-size-2", type=int, default=8)
    parser.add_argument("--eegnet-dropout-rate", type=float, default=0.25)
    parser.add_argument("--eegnet-l2-weight", type=float, default=1e-4)
    parser.add_argument("--triplet-loss-weight", type=float, default=1.0)
    parser.add_argument("--triplet-margin", type=float, default=0.1)
    parser.add_argument(
        "--triplet-mining-strategy",
        type=str,
        default="batch_hard",
        choices=("batch_hard", "batch_all", "triplet_center"),
    )
    parser.add_argument("--triplet-center-gradient-clip-norm", type=float, default=0.01)
    parser.add_argument("--gaussian-noise-std", type=float, default=0.01)
    parser.add_argument(
        "--deterministic-ops",
        action="store_true",
        help="Enable deterministic TensorFlow ops (slower but reproducible).",
    )
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--tasks-per-epoch", type=int, default=1)
    parser.add_argument("--task-batch-size", type=int, default=1)
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=1,
        help="Number of episodic tasks whose samples are encoded together.",
    )
    parser.add_argument("--val-tasks", type=int, default=1)
    parser.add_argument("--heldout-eval-tasks", type=int, default=1)
    parser.add_argument(
        "--subject-eval-tasks",
        type=int,
        default=None,
        help="Deprecated alias for --heldout-eval-tasks.",
    )
    parser.add_argument("--k-shot-adaptation-steps", type=int, default=10)
    parser.add_argument("--train-log-every", type=int, default=10)
    parser.add_argument("--eval-log-every", type=int, default=5)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--val-every-n-train-steps", type=int, default=20)
    parser.add_argument(
        "--validation-checkpoint-metric",
        type=str,
        default="accuracy",
        choices=SUPPORTED_VALIDATION_CHECKPOINT_METRICS,
        help="Validation metric used to select the fold model for held-out eval.",
    )
    parser.add_argument(
        "--validation-checkpoint-mode",
        type=str,
        default="auto",
        choices=VALIDATION_CHECKPOINT_MODES,
        help="Direction for the validation checkpoint metric.",
    )
    parser.add_argument("--summary-every-n-train-steps", type=int, default=20)
    parser.add_argument("--train-prefetch-batches", type=int, default=2)
    parser.add_argument(
        "--train-progress-write-every-n-batches",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--csv-flush-every-events",
        type=int,
        default=100,
    )
    parser.add_argument("--disable-window-shift", action="store_true")
    parser.add_argument(
        "--logging-verbosity",
        type=int,
        default=1,
        choices=(0, 1, 2),
        help="0=minimal, 1=standard, 2=detailed logging",
    )
    parser.add_argument(
        "--training-progress-output-dir",
        type=str,
        default="outputs/training_progress",
    )
    parser.add_argument(
        "--model-architecture-output",
        type=str,
        default="outputs/model_architecture/model_summary.txt",
    )
    parser.add_argument("--skip-model-architecture-save", action="store_true")
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Optional debug limit. Omit for full LOSO over all subjects.",
    )
    parser.add_argument(
        "--loso-start-index",
        type=int,
        default=None,
        help="1-based inclusive LOSO fold index to start from.",
    )
    parser.add_argument(
        "--loso-stop-index",
        type=int,
        default=None,
        help="1-based inclusive LOSO fold index to stop at.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/full_loso/full_loso_results.json",
    )
    args = parser.parse_args()
    if args.subject_eval_tasks is not None:
        args.heldout_eval_tasks = int(args.subject_eval_tasks)

    payload = run_full_loso_trial(args)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
