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
    SOURCE_SUBJECT_PROTOTYPE_VOTE_SOFTMAX_SCOPES,
    SUPPORTED_VALIDATION_CHECKPOINT_METRICS,
    SUPPORTED_DATASET_SOURCES,
    VALIDATION_CHECKPOINT_MODES,
)
from learner.few_shot_pain_learner import FewShotPainLearner
from scripts.analyze_matched_query_personalization import run_analysis
from utils.logger import setup_logger


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return values


def _default_task_class_ids(dataset_source: str) -> tuple[int, ...]:
    if str(dataset_source).strip().lower() == "senseemotion":
        return (0, 1, 2, 3)
    return (0, 4)


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
        "source_subject_prototype_vote_accuracy": _metric_summary(
            cv_results.get("source_subject_prototype_vote_accuracies", [])
        ),
        "source_subject_prototype_vote_loss": _metric_summary(
            cv_results.get("source_subject_prototype_vote_losses", [])
        ),
        "source_subject_prototype_vote_f1": _metric_summary(
            cv_results.get("source_subject_prototype_vote_f1s", [])
        ),
    }
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
    dataset_source = str(getattr(args, "dataset_source", "biovid_part_a"))
    task_class_ids = (
        _default_task_class_ids(dataset_source)
        if args.task_class_ids is None
        else _parse_int_tuple(args.task_class_ids)
    )

    logger.info("Stage 1/5: Building run configuration")
    config = PainDatasetConfig(
        seed=args.seed,
        deterministic_ops=bool(args.deterministic_ops),
        dataset_source=dataset_source,
        data_variant=args.data_variant,
        task_class_ids=task_class_ids,
        k_shot=args.k_shot,
        q_query=args.q_query,
        task_normalize_mode=args.normalize_mode,
        task_construction_mode=args.task_construction_mode,
        attention_mode="can",
        can_attention_temperature=float(
            getattr(args, "can_attention_temperature", 1.0)
        ),
        can_meta_hidden_dim=int(getattr(args, "can_meta_hidden_dim", 32)),
        can_local_loss_weight=float(getattr(args, "can_local_loss_weight", 1.0)),
        can_margin_loss_weight=float(getattr(args, "can_margin_loss_weight", 0.2)),
        can_margin_target=float(getattr(args, "can_margin_target", 0.3)),
        contrastive_loss_weight=float(
            getattr(args, "contrastive_loss_weight", 0.0)
        ),
        contrastive_temperature=float(
            getattr(args, "contrastive_temperature", 0.1)
        ),
        contrastive_cross_subject=bool(
            getattr(args, "contrastive_cross_subject", True)
        ),
        can_support_mode=str(
            getattr(args, "can_support_mode", "learned_prototype_memory")
        ),
        learned_prototype_slots_per_class=int(
            getattr(args, "learned_prototype_slots_per_class", 2)
        ),
        prototype_bank_init_samples_per_class=int(
            getattr(args, "prototype_bank_init_samples_per_class", 0)
        ),
        prototype_finetune_epochs=int(getattr(args, "prototype_finetune_epochs", 1)),
        prototype_finetune_tasks_per_epoch=getattr(
            args,
            "prototype_finetune_tasks_per_epoch",
            50,
        ),
        prototype_phase2_loss_mode=str(
            getattr(args, "prototype_phase2_loss_mode", "ce_can")
        ),
        source_subject_prototype_vote_enabled=bool(
            getattr(
                args,
                "source_subject_prototype_vote_enabled",
                not bool(
                    getattr(args, "disable_source_subject_prototype_vote", False)
                ),
            )
        ),
        source_subject_prototype_vote_use_base_index=bool(
            getattr(args, "source_subject_prototype_vote_use_base_index", True)
        ),
        source_subject_prototype_vote_query_normalize_with_subject_stats=bool(
            getattr(
                args,
                "source_subject_prototype_vote_query_normalize_with_subject_stats",
                True,
            )
        ),
        source_subject_prototype_vote_softmax_scope=str(
            getattr(args, "source_subject_prototype_vote_softmax_scope", "global")
        ),
        train_batch_size=args.task_batch_size,
        task_chunk_size=max(1, int(getattr(args, "task_chunk_size", 1))),
        tasks_per_epoch=max(1, int(args.tasks_per_epoch)),
        val_tasks=max(1, int(args.val_tasks)),
        heldout_eval_tasks=max(1, int(args.heldout_eval_tasks)),
        matched_query_eval=bool(args.matched_query_eval),
        matched_query_support_repeats=max(
            1, int(args.matched_query_support_repeats)
        ),
        num_epochs=max(1, int(args.num_epochs)),
        k_shot_adaptation_steps=max(0, int(args.k_shot_adaptation_steps)),
        train_log_every=max(1, int(args.train_log_every)),
        eval_log_every=max(1, int(args.eval_log_every)),
        val_batch_size=max(1, int(args.val_batch_size)),
        val_every_n_train_steps=max(1, int(args.val_every_n_train_steps)),
        disable_validation=bool(getattr(args, "disable_validation", False)),
        validation_checkpoint_metric=str(
            getattr(args, "validation_checkpoint_metric", "accuracy")
        ),
        validation_checkpoint_mode=str(
            getattr(args, "validation_checkpoint_mode", "auto")
        ),
        lr_schedule=str(getattr(args, "lr_schedule", "constant")),
        lr_decay_alpha=float(getattr(args, "lr_decay_alpha", 0.1)),
        train_prefetch_batches=max(1, int(getattr(args, "train_prefetch_batches", 2))),
        train_progress_write_every_n_batches=max(
            1, int(args.train_progress_write_every_n_batches)
        ),
        csv_flush_every_events=max(1, int(args.csv_flush_every_events)),
        single_loso_fold=False,  # Full LOSO over all available subjects.
        loso_start_index=args.loso_start_index,
        loso_stop_index=args.loso_stop_index,
        eegnet_temporal_filters=args.eegnet_temporal_filters,
        eegnet_depth_multiplier=args.eegnet_depth_multiplier,
        eegnet_separable_filters=args.eegnet_separable_filters,
        eegnet_temporal_kernel_size=args.eegnet_temporal_kernel_size,
        eegnet_separable_kernel_size=args.eegnet_separable_kernel_size,
        eegnet_pool_size_1=args.eegnet_pool_size_1,
        eegnet_pool_size_2=args.eegnet_pool_size_2,
        eegnet_dropout_rate=args.eegnet_dropout_rate,
        eegnet_l2_weight=args.eegnet_l2_weight,
        eegnet_normalization=str(getattr(args, "eegnet_normalization", "group")),
        eegnet_group_norm_groups=int(
            getattr(args, "eegnet_group_norm_groups", 4)
        ),
        can_local_pool_temperature=float(
            getattr(args, "can_local_pool_temperature", 0.1)
        ),
        encoder_backend=str(getattr(args, "encoder_backend", "eegnet")),
        crossmod_num_heads=int(getattr(args, "crossmod_num_heads", 8)),
        crossmod_hidden_dim=int(getattr(args, "crossmod_hidden_dim", 128)),
        crossmod_num_layers=int(getattr(args, "crossmod_num_layers", 2)),
        crossmod_positional_base=float(
            getattr(args, "crossmod_positional_base", 10000.0)
        ),
        crossmod_attention_dropout_rate=float(
            getattr(args, "crossmod_attention_dropout_rate", 0.0)
        ),
        crossmod_ff_activation=str(getattr(args, "crossmod_ff_activation", "relu")),
        enable_window_shift_augmentation=not args.disable_window_shift,
        gaussian_noise_std=args.gaussian_noise_std,
        logging_verbosity=args.logging_verbosity,
        disable_training_logging=bool(
            getattr(args, "disable_training_logging", False)
        ),
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
    matched_query_summary = None
    if config.matched_query_eval:
        matched_query_summary = run_analysis(
            cv_results["matched_query_repeat_metric_files"],
            args.matched_query_analysis_output_dir,
            bootstrap_trials=20_000,
            confidence=0.95,
            seed=int(config.seed),
        )
        summary["matched_query_primary"] = matched_query_summary

    config_payload = {
        "seed": int(config.seed),
        "dataset_source": str(config.dataset_source),
        "data_variant": str(config.data_variant),
        "task_class_ids": list(config.task_class_ids),
        "k_shot": int(config.k_shot),
        "q_query": int(config.q_query),
        "task_construction_mode": str(config.task_construction_mode),
        "normalize_mode": str(config.task_normalize_mode),
        "attention_mode": str(config.attention_mode),
        "can_attention_temperature": float(config.can_attention_temperature),
        "can_meta_hidden_dim": int(config.can_meta_hidden_dim),
        "can_local_loss_weight": float(config.can_local_loss_weight),
        "can_margin_loss_weight": float(config.can_margin_loss_weight),
        "can_margin_target": float(config.can_margin_target),
        "contrastive_loss_weight": float(config.contrastive_loss_weight),
        "contrastive_temperature": float(config.contrastive_temperature),
        "contrastive_cross_subject": bool(config.contrastive_cross_subject),
        "can_support_mode": str(config.can_support_mode),
        "learned_prototype_slots_per_class": int(
            config.learned_prototype_slots_per_class
        ),
        "prototype_bank_init_samples_per_class": int(
            config.prototype_bank_init_samples_per_class
        ),
        "prototype_finetune_epochs": int(config.prototype_finetune_epochs),
        "prototype_finetune_tasks_per_epoch": (
            int(config.prototype_finetune_tasks_per_epoch)
            if config.prototype_finetune_tasks_per_epoch is not None
            else None
        ),
        "prototype_phase2_loss_mode": str(config.prototype_phase2_loss_mode),
        "source_subject_prototype_vote_enabled": bool(
            config.source_subject_prototype_vote_enabled
        ),
        "source_subject_prototype_vote_use_base_index": bool(
            config.source_subject_prototype_vote_use_base_index
        ),
        "source_subject_prototype_vote_query_normalize_with_subject_stats": bool(
            config.source_subject_prototype_vote_query_normalize_with_subject_stats
        ),
        "source_subject_prototype_vote_softmax_scope": str(
            config.source_subject_prototype_vote_softmax_scope
        ),
        "learning_rate": float(args.learning_rate),
        "lr_schedule": str(config.lr_schedule),
        "lr_decay_alpha": float(config.lr_decay_alpha),
        "encoder_backend": str(config.encoder_backend),
        "eegnet_temporal_filters": int(config.eegnet_temporal_filters),
        "eegnet_depth_multiplier": int(config.eegnet_depth_multiplier),
        "eegnet_separable_filters": int(config.eegnet_separable_filters),
        "eegnet_temporal_kernel_size": int(config.eegnet_temporal_kernel_size),
        "eegnet_separable_kernel_size": int(config.eegnet_separable_kernel_size),
        "eegnet_pool_size_1": int(config.eegnet_pool_size_1),
        "eegnet_pool_size_2": int(config.eegnet_pool_size_2),
        "eegnet_dropout_rate": float(config.eegnet_dropout_rate),
        "eegnet_l2_weight": float(config.eegnet_l2_weight),
        "crossmod_num_heads": int(config.crossmod_num_heads),
        "crossmod_hidden_dim": int(config.crossmod_hidden_dim),
        "crossmod_num_layers": int(config.crossmod_num_layers),
        "crossmod_positional_base": float(config.crossmod_positional_base),
        "crossmod_attention_dropout_rate": float(
            config.crossmod_attention_dropout_rate
        ),
        "crossmod_ff_activation": str(config.crossmod_ff_activation),
        "num_epochs": int(config.num_epochs),
        "tasks_per_epoch": int(config.tasks_per_epoch),
        "train_batch_size": int(config.train_batch_size),
        "task_chunk_size": int(config.task_chunk_size),
        "val_tasks": int(config.val_tasks),
        "heldout_eval_tasks": int(config.heldout_eval_tasks),
        "matched_query_eval": bool(config.matched_query_eval),
        "matched_query_support_repeats": int(
            config.matched_query_support_repeats
        ),
        "matched_query_analysis_output_dir": str(
            args.matched_query_analysis_output_dir
        ),
        "validation_checkpoint_metric": str(config.validation_checkpoint_metric),
        "validation_checkpoint_mode": str(config.validation_checkpoint_mode),
        "disable_validation": bool(config.disable_validation),
        "disable_training_logging": bool(config.disable_training_logging),
        "k_shot_adaptation_steps": int(config.k_shot_adaptation_steps),
        "window_shift_enabled": bool(config.enable_window_shift_augmentation),
        "gaussian_noise_std": float(config.gaussian_noise_std),
        "deterministic_ops": bool(config.deterministic_ops),
        "train_progress_write_every_n_batches": int(
            config.train_progress_write_every_n_batches
        ),
        "csv_flush_every_events": int(config.csv_flush_every_events),
        "train_prefetch_batches": int(config.train_prefetch_batches),
        "loso_start_index": config.loso_start_index,
        "loso_stop_index": config.loso_stop_index,
        "max_folds": int(args.max_folds) if args.max_folds is not None else None,
    }

    payload: dict[str, Any] = {
        "script": "tests/full_loso_trial.py",
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "data_dir": args.data_dir,
        "config": config_payload,
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
            "Run full LOSO training/evaluation with learned CAN prototype memory "
            "across all held-out subjects and store results as JSON."
        )
    )
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument(
        "--dataset-source",
        type=str,
        default="biovid_part_a",
        choices=SUPPORTED_DATASET_SOURCES,
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
    parser.add_argument(
        "--task-class-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated raw class ids. Defaults to 0,4 for BioVid/PainMonit "
            "and 0,1,2,3 for SenseEmotion."
        ),
    )
    parser.add_argument(
        "--task-construction-mode",
        type=str,
        default="single_subject",
        choices=("single_subject", "cross_subject", "mixed"),
    )
    parser.add_argument("--can-attention-temperature", type=float, default=1.0)
    parser.add_argument("--can-meta-hidden-dim", type=int, default=32)
    parser.add_argument("--can-local-loss-weight", type=float, default=1.0)
    parser.add_argument("--can-margin-loss-weight", type=float, default=0.2)
    parser.add_argument("--can-margin-target", type=float, default=0.3)
    parser.add_argument(
        "--can-support-mode",
        type=str,
        default="learned_prototype_memory",
        choices=("sampled", "learned_prototype_memory"),
    )
    parser.add_argument("--learned-prototype-slots-per-class", type=int, default=2)
    parser.add_argument("--prototype-bank-init-samples-per-class", type=int, default=86*3)
    parser.add_argument("--prototype-finetune-epochs", type=int, default=1)
    parser.add_argument("--prototype-finetune-tasks-per-epoch", type=int, default=50)
    parser.add_argument(
        "--prototype-phase2-loss-mode",
        type=str,
        default="ce_can",
        choices=("ce_can",),
    )
    parser.add_argument(
        "--disable-source-subject-prototype-vote",
        action="store_true",
        help="Disable post-training source-subject prototype vote inference.",
    )
    parser.add_argument(
        "--source-subject-prototype-vote-use-base-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use base, non-augmented source samples when building vote prototypes.",
    )
    parser.add_argument(
        "--source-subject-prototype-vote-query-normalize-with-subject-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize held-out vote queries with held-out subject statistics.",
    )
    parser.add_argument(
        "--source-subject-prototype-vote-softmax-scope",
        type=str,
        default="global",
        choices=SOURCE_SUBJECT_PROTOTYPE_VOTE_SOFTMAX_SCOPES,
    )
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="support",
        choices=("subject", "split", "support", "none"),
    )
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=("constant", "cosine"),
        help="Learning-rate schedule passed to the Keras optimizer.",
    )
    parser.add_argument(
        "--lr-decay-alpha",
        type=float,
        default=0.1,
        help="Final LR fraction for cosine decay.",
    )
    parser.add_argument("--eegnet-temporal-filters", type=int, default=8)
    parser.add_argument("--eegnet-depth-multiplier", type=int, default=2)
    parser.add_argument("--eegnet-separable-filters", type=int, default=16)
    parser.add_argument("--eegnet-temporal-kernel-size", type=int, default=64)
    parser.add_argument("--eegnet-separable-kernel-size", type=int, default=16)
    parser.add_argument("--eegnet-pool-size-1", type=int, default=4)
    parser.add_argument("--eegnet-pool-size-2", type=int, default=8)
    parser.add_argument("--eegnet-dropout-rate", type=float, default=0.25)
    parser.add_argument("--eegnet-l2-weight", type=float, default=1e-4)
    parser.add_argument(
        "--encoder-backend",
        type=str,
        default="crossmod",
        choices=("eegnet", "crossmod"),
        help="Encoder backend. 'crossmod' is a CAN-only EDA/ECG feature-map encoder.",
    )
    parser.add_argument("--crossmod-num-heads", type=int, default=8)
    parser.add_argument("--crossmod-hidden-dim", type=int, default=128)
    parser.add_argument("--crossmod-num-layers", type=int, default=2)
    parser.add_argument("--crossmod-positional-base", type=float, default=10000.0)
    parser.add_argument("--crossmod-attention-dropout-rate", type=float, default=0.0)
    parser.add_argument("--crossmod-ff-activation", type=str, default="relu")
    parser.add_argument("--gaussian-noise-std", type=float, default=0.01)
    parser.add_argument(
        "--deterministic-ops",
        action="store_true",
        help="Enable deterministic TensorFlow ops (slower but reproducible).",
    )
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--tasks-per-epoch", type=int, default=1)
    parser.add_argument("--task-batch-size", type=int, default=16)
    parser.add_argument(
        "--task-chunk-size",
        type=int,
        default=1,
        help="Number of episodic tasks whose samples are encoded together.",
    )
    parser.add_argument("--val-tasks", type=int, default=1)
    parser.add_argument("--heldout-eval-tasks", type=int, default=1)
    parser.add_argument(
        "--matched-query-eval",
        action="store_true",
        help=(
            "Run the primary matched-query comparison using held-out Train support, "
            "fixed held-out Test queries, and fold-source normalization. Requires "
            "--normalize-mode split and zero adaptation steps."
        ),
    )
    parser.add_argument(
        "--matched-query-support-repeats",
        type=int,
        default=500,
        help="Independent target-support draws per held-out subject.",
    )
    parser.add_argument(
        "--matched-query-analysis-output-dir",
        type=str,
        default="outputs/matched_query_personalization",
    )
    parser.add_argument(
        "--subject-eval-tasks",
        type=int,
        default=None,
        help="Deprecated alias for --heldout-eval-tasks.",
    )
    parser.add_argument("--k-shot-adaptation-steps", type=int, default=0)
    parser.add_argument("--train-log-every", type=int, default=10)
    parser.add_argument("--eval-log-every", type=int, default=5)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--val-every-n-train-steps", type=int, default=20)
    parser.add_argument(
        "--disable-validation",
        action="store_true",
        help="Skip validation evaluation and use the final training state.",
    )
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
        "--disable-training-logging",
        action="store_true",
        help="Skip training progress log messages and train-update CSV rows.",
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
