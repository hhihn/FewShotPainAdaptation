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

from data_loaders.pain_ds_config import PainDatasetConfig
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
    return {
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
    filters_list = _parse_int_tuple(args.filters)
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
        train_batch_size=args.task_batch_size,
        tasks_per_epoch=max(1, int(args.tasks_per_epoch)),
        val_tasks=max(1, int(args.val_tasks)),
        heldout_eval_tasks=max(1, int(args.heldout_eval_tasks)),
        num_epochs=max(1, int(args.num_epochs)),
        k_shot_adaptation_steps=max(0, int(args.k_shot_adaptation_steps)),
        train_log_every=max(1, int(args.train_log_every)),
        eval_log_every=max(1, int(args.eval_log_every)),
        val_batch_size=max(1, int(args.val_batch_size)),
        val_every_n_train_steps=max(1, int(args.val_every_n_train_steps)),
        summary_every_n_train_steps=max(1, int(args.summary_every_n_train_steps)),
        train_prefetch_batches=max(1, int(getattr(args, "train_prefetch_batches", 2))),
        train_progress_write_every_n_batches=max(
            1, int(args.train_progress_write_every_n_batches)
        ),
        csv_flush_every_events=max(1, int(args.csv_flush_every_events)),
        single_loso_fold=False,  # Full LOSO over all available subjects.
        loso_start_index=args.loso_start_index,
        loso_stop_index=args.loso_stop_index,
        embedding_dim=args.embedding_dim,
        num_tcn_blocks=len(filters_list),
        filters_list=filters_list,
        tcn_attention_heads=args.tcn_attention_heads,
        tcn_attention_key_dim=args.tcn_attention_key_dim,
        tcn_attention_dropout=args.tcn_attention_dropout,
        tcn_transformer_layers=args.tcn_transformer_layers,
        tcn_transformer_ffn_dim=args.tcn_transformer_ffn_dim,
        tcn_attention_pool_size=args.tcn_attention_pool_size,
        use_attention=bool(getattr(args, "use_attention", True)),
        supcon_loss_weight=float(getattr(args, "supcon_loss_weight", 0.0)),
        supcon_temperature=float(getattr(args, "supcon_temperature", 0.05)),
        triplet_loss_weight=float(getattr(args, "triplet_loss_weight", 1.0)),
        triplet_margin=float(getattr(args, "triplet_margin", 0.2)),
        triplet_mining_strategy=str(
            getattr(args, "triplet_mining_strategy", "batch_hard")
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
        fusion_method=args.fusion_method,
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
            "fusion_method": str(args.fusion_method),
            "learning_rate": float(args.learning_rate),
            "embedding_dim": int(config.embedding_dim),
            "filters": list(filters_list),
            "tcn_dilation_rates": list(config.tcn_dilation_rates),
            "tcn_attention_heads": int(config.tcn_attention_heads),
            "tcn_attention_key_dim": int(config.tcn_attention_key_dim),
            "tcn_transformer_layers": int(config.tcn_transformer_layers),
            "tcn_transformer_ffn_dim": int(config.tcn_transformer_ffn_dim),
            "tcn_attention_dropout": float(config.tcn_attention_dropout),
            "use_attention": bool(config.use_attention),
            "num_epochs": int(config.num_epochs),
            "tasks_per_epoch": int(config.tasks_per_epoch),
            "train_batch_size": int(config.train_batch_size),
            "val_tasks": int(config.val_tasks),
            "heldout_eval_tasks": int(config.heldout_eval_tasks),
            "k_shot_adaptation_steps": int(config.k_shot_adaptation_steps),
            "window_shift_enabled": bool(config.enable_window_shift_augmentation),
            "gaussian_noise_std": float(config.gaussian_noise_std),
            "supcon_loss_weight": float(config.supcon_loss_weight),
            "supcon_temperature": float(config.supcon_temperature),
            "triplet_loss_weight": float(config.triplet_loss_weight),
            "triplet_margin": float(config.triplet_margin),
            "triplet_mining_strategy": str(config.triplet_mining_strategy),
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
        "--fusion-method",
        type=str,
        default="mean",
        choices=("mean", "gated", "transformer_ib"),
    )
    parser.add_argument(
        "--classifier-mode",
        type=str,
        default="prototype",
        choices=("prototype", "soft_knn"),
    )
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="support",
        choices=("subject", "split", "support", "none"),
    )
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--filters", type=str, default="16,32,64,128")
    parser.add_argument("--tcn-attention-heads", type=int, default=8)
    parser.add_argument("--tcn-attention-key-dim", type=int, default=8)
    parser.add_argument("--tcn-attention-dropout", type=float, default=0.1)
    parser.add_argument("--tcn-transformer-layers", type=int, default=2)
    parser.add_argument("--tcn-transformer-ffn-dim", type=int, default=256)
    parser.add_argument("--tcn-attention-pool-size", type=int, default=0)
    parser.add_argument("--use-attention", action="store_true")
    parser.add_argument("--supcon-loss-weight", type=float, default=0.7)
    parser.add_argument("--supcon-temperature", type=float, default=0.05)
    parser.add_argument("--triplet-loss-weight", type=float, default=1.0)
    parser.add_argument("--triplet-margin", type=float, default=0.1)
    parser.add_argument(
        "--triplet-mining-strategy",
        type=str,
        default="batch_hard",
        choices=("batch_hard", "batch_all"),
    )
    parser.add_argument("--gaussian-noise-std", type=float, default=0.01)
    parser.add_argument(
        "--deterministic-ops",
        action="store_true",
        help="Enable deterministic TensorFlow ops (slower but reproducible).",
    )
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--tasks-per-epoch", type=int, default=1)
    parser.add_argument("--task-batch-size", type=int, default=1)
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
