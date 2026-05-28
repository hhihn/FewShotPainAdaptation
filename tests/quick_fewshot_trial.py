import argparse
import csv
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


def _sample_tasks(sampler, num_tasks: int) -> list[dict[str, np.ndarray]]:
    return [sampler.get_task() for _ in range(max(1, int(num_tasks)))]


def _sample_subject_tasks(sampler, num_tasks: int) -> list[dict[str, np.ndarray]]:
    subjects = [int(subject) for subject in getattr(sampler, "active_subjects", [])]
    if not subjects:
        return _sample_tasks(sampler, num_tasks)
    return [
        sampler.get_task(subject=subjects[task_idx % len(subjects)])
        for task_idx in range(max(1, int(num_tasks)))
    ]


def _evaluate_bank(
    learner: FewShotPainLearner, tasks: list[dict[str, np.ndarray]]
) -> dict[str, float]:
    loss, accuracy, contrastive_loss, triplet_loss = learner.evaluate_batch_step(tasks)
    metrics = {
        "loss": float(loss.numpy()),
        "accuracy": float(accuracy.numpy()),
    }
    if getattr(learner, "attention_mode", "none") != "can":
        metrics.update(
            {
                "contrastive_loss": float(contrastive_loss.numpy()),
                "triplet_loss": float(triplet_loss.numpy()),
            }
        )
    return metrics


def _log_trial_summary(
    logger,
    prefix: str,
    composite_accuracy: float,
    train_accuracy: float,
    val_accuracy: float,
    heldout_accuracy: float,
    elapsed_seconds: float,
) -> None:
    logger.info(
        f"{prefix}: "
        f"composite={composite_accuracy:.4f}, "
        f"train_acc={train_accuracy:.4f}, "
        f"val_acc={val_accuracy:.4f}, "
        f"heldout_acc={heldout_accuracy:.4f}, "
        f"elapsed_seconds={elapsed_seconds:.2f}"
    )


def _cyclic_task_batch(
    tasks: list[dict[str, np.ndarray]],
    update_idx: int,
    task_batch_size: int,
) -> list[dict[str, np.ndarray]]:
    start = update_idx * task_batch_size
    return [tasks[(start + offset) % len(tasks)] for offset in range(task_batch_size)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_payload = {
        key: json.dumps(value, sort_keys=True)
        if isinstance(value, (dict, list))
        else value
        for key, value in payload.items()
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_payload.keys()))
        writer.writeheader()
        writer.writerow(flat_payload)


def _with_repeat_suffix(path: str, repeat_idx: int, repeats: int) -> str:
    """Append a repeat suffix to an output path when running multiple repeats."""
    if repeats <= 1:
        return path
    raw_path = Path(path)
    suffix = f"_repeat_{repeat_idx + 1:02d}"
    return str(raw_path.with_name(f"{raw_path.stem}{suffix}{raw_path.suffix}"))


def _run_single_quick_trial(args: argparse.Namespace) -> dict[str, Any]:
    logger = setup_logger("quick_fewshot_trial")
    if args.logging_verbosity <= 0:
        logger.setLevel(30)
    elif args.logging_verbosity == 1:
        logger.setLevel(20)
    else:
        logger.setLevel(10)
    start_time = time.perf_counter()

    config = PainDatasetConfig(
        seed=args.seed,
        deterministic_ops=True,
        task_class_ids=_parse_int_tuple(args.task_class_ids),
        k_shot=args.k_shot,
        q_query=args.q_query,
        task_normalize_mode=args.normalize_mode,
        classifier_mode=args.classifier_mode,
        attention_mode=str(getattr(args, "attention_mode", "none")),
        can_attention_temperature=float(
            getattr(args, "can_attention_temperature", 1.0)
        ),
        can_meta_hidden_dim=int(getattr(args, "can_meta_hidden_dim", 32)),
        can_local_loss_weight=float(getattr(args, "can_local_loss_weight", 1.0)),
        can_global_loss_weight=float(getattr(args, "can_global_loss_weight", 0.1)),
        can_margin_loss_weight=float(getattr(args, "can_margin_loss_weight", 0.2)),
        can_margin_target=float(getattr(args, "can_margin_target", 0.3)),
        train_batch_size=args.task_batch_size,
        embedding_batch_size=max(1, int(getattr(args, "embedding_batch_size", 1))),
        tasks_per_epoch=max(1, args.updates * args.task_batch_size),
        val_tasks=max(1, args.val_tasks),
        heldout_eval_tasks=max(1, args.heldout_tasks),
        num_epochs=1,
        single_loso_fold=True,
        train_prefetch_batches=max(1, int(getattr(args, "train_prefetch_batches", 2))),
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
        lr_schedule=str(getattr(args, "lr_schedule", "constant")),
        lr_decay_alpha=float(getattr(args, "lr_decay_alpha", 0.1)),
        enable_window_shift_augmentation=not args.disable_window_shift,
        gaussian_noise_std=args.gaussian_noise_std,
        logging_verbosity=args.logging_verbosity,
    )
    learner = FewShotPainLearner(
        config=config,
        data_dir=args.data_dir,
        learning_rate=args.learning_rate,
    )

    held_out_subject = (
        int(args.held_out_subject)
        if args.held_out_subject is not None
        else int(learner.cv.subjects[0])
    )
    fold = learner.cv.get_fold(held_out_subject)

    train_tasks = _sample_subject_tasks(
        fold["train_sampler"],
        max(1, args.updates * args.task_batch_size),
    )
    model_architecture_file = None
    if args.model_architecture_output and train_tasks:
        model_architecture_file = learner._save_model_architecture(
            sample_task=train_tasks[0],
            output_path=args.model_architecture_output,
        )
    train_eval_tasks = train_tasks[
        : max(1, min(args.train_eval_tasks, len(train_tasks)))
    ]
    val_tasks = _sample_subject_tasks(fold["val_sampler"], args.val_tasks)
    heldout_tasks = _sample_tasks(fold["test_sampler"], args.heldout_tasks)

    before_train = _evaluate_bank(learner, train_eval_tasks)
    before_val = _evaluate_bank(learner, val_tasks)
    before_heldout = _evaluate_bank(learner, heldout_tasks)
    if args.logging_verbosity >= 1:
        _log_trial_summary(
            logger,
            "Quick few-shot trial baseline",
            float(
                np.mean(
                    [
                        before_train["accuracy"],
                        before_val["accuracy"],
                        before_heldout["accuracy"],
                    ]
                )
            ),
            before_train["accuracy"],
            before_val["accuracy"],
            before_heldout["accuracy"],
            time.perf_counter() - start_time,
        )

    update_history = []
    total_updates = max(1, args.updates)
    summary_every = max(1, int(args.summary_every_n_updates))
    for update_idx in range(max(1, args.updates)):
        update_start = time.perf_counter()
        task_batch = _cyclic_task_batch(train_tasks, update_idx, args.task_batch_size)
        (
            loss,
            task_loss,
            accuracy,
            contrastive_loss,
            triplet_loss,
            can_local_loss,
            can_global_loss,
            can_margin_loss,
        ) = learner.train_batch_step(task_batch)
        elapsed = time.perf_counter() - start_time
        avg_update_time = elapsed / max(1, update_idx + 1)
        (total_updates - (update_idx + 1)) * avg_update_time
        update_history.append(
            {
                "update": update_idx + 1,
                "loss": float(loss.numpy()),
                "task_loss": float(task_loss.numpy()),
                "accuracy": float(accuracy.numpy()),
                "can_local_loss": float(can_local_loss.numpy()),
                "can_margin_loss": float(can_margin_loss.numpy()),
                "elapsed_seconds": float(time.perf_counter() - update_start),
            }
        )
        if str(config.attention_mode).lower() != "can":
            update_history[-1].update(
                {
                    "contrastive_loss": float(contrastive_loss.numpy()),
                    "triplet_loss": float(triplet_loss.numpy()),
                    "can_global_loss": float(can_global_loss.numpy()),
                }
            )

        if (update_idx + 1) % summary_every == 0 or (update_idx + 1) == total_updates:
            after_train = _evaluate_bank(learner, train_eval_tasks)
            after_val = _evaluate_bank(learner, val_tasks)
            after_heldout = _evaluate_bank(learner, heldout_tasks)
            composite_accuracy = float(
                np.mean(
                    [
                        after_train["accuracy"],
                        after_val["accuracy"],
                        after_heldout["accuracy"],
                    ]
                )
            )
            _log_trial_summary(
                logger,
                f"Quick few-shot trial after {update_idx + 1} updates",
                composite_accuracy,
                after_train["accuracy"],
                after_val["accuracy"],
                after_heldout["accuracy"],
                time.perf_counter() - start_time,
            )

    after_train = _evaluate_bank(learner, train_eval_tasks)
    after_val = _evaluate_bank(learner, val_tasks)
    after_heldout = _evaluate_bank(learner, heldout_tasks)
    elapsed_seconds = time.perf_counter() - start_time

    final_composite_accuracy = float(
        np.mean(
            [
                after_train["accuracy"],
                after_val["accuracy"],
                after_heldout["accuracy"],
            ]
        )
    )
    heldout_generalization_gap = float(
        after_train["accuracy"] - after_heldout["accuracy"]
    )

    payload: dict[str, Any] = {
        "script": "tests/quick_fewshot_trial.py",
        "elapsed_seconds": elapsed_seconds,
        "seed": args.seed,
        "held_out_subject": held_out_subject,
        "train_subject_count": int(fold["n_train_subjects"]),
        "val_subject_count": int(fold["n_val_subjects"]),
        "updates": max(1, args.updates),
        "task_batch_size": args.task_batch_size,
        "embedding_batch_size": int(config.embedding_batch_size),
        "train_task_count": len(train_tasks),
        "train_eval_task_count": len(train_eval_tasks),
        "val_task_count": len(val_tasks),
        "heldout_task_count": len(heldout_tasks),
        "k_shot": args.k_shot,
        "q_query": args.q_query,
        "task_class_ids": list(config.task_class_ids),
        "classifier_mode": args.classifier_mode,
        "attention_mode": str(config.attention_mode),
        "can_attention_temperature": float(config.can_attention_temperature),
        "can_meta_hidden_dim": int(config.can_meta_hidden_dim),
        "can_local_loss_weight": float(config.can_local_loss_weight),
        "can_global_loss_weight": float(config.can_global_loss_weight),
        "can_margin_loss_weight": float(config.can_margin_loss_weight),
        "can_margin_target": float(config.can_margin_target),
        "learning_rate": float(args.learning_rate),
        "lr_schedule": str(config.lr_schedule),
        "lr_decay_alpha": float(config.lr_decay_alpha),
        "normalize_mode": args.normalize_mode,
        "embedding_dim": args.embedding_dim,
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
        "window_shift_enabled": bool(config.enable_window_shift_augmentation),
        "gaussian_noise_std": float(config.gaussian_noise_std),
        "triplet_loss_weight": float(config.triplet_loss_weight),
        "triplet_margin": float(config.triplet_margin),
        "triplet_mining_strategy": str(config.triplet_mining_strategy),
        "triplet_center_gradient_clip_norm": float(
            config.triplet_center_gradient_clip_norm
        ),
        "model_architecture_file": model_architecture_file,
        "before": {
            "train": before_train,
            "val": before_val,
            "heldout": before_heldout,
        },
        "after": {
            "train": after_train,
            "val": after_val,
            "heldout": after_heldout,
        },
        "update_history": update_history,
        "final_composite_accuracy": final_composite_accuracy,
        "heldout_generalization_gap": heldout_generalization_gap,
        "train_accuracy_delta": float(
            after_train["accuracy"] - before_train["accuracy"]
        ),
        "val_accuracy_delta": float(after_val["accuracy"] - before_val["accuracy"]),
        "heldout_accuracy_delta": float(
            after_heldout["accuracy"] - before_heldout["accuracy"]
        ),
    }

    if str(config.attention_mode).lower() == "can":
        payload.update(
            {
                "embedding_projection_enabled": False,
                "can_global_loss_weight": 0.0,
            }
        )
        for key in (
            "can_global_loss_weight",
            "embedding_batch_size",
            "embedding_dim",
            "triplet_loss_weight",
            "triplet_margin",
            "triplet_mining_strategy",
            "triplet_center_gradient_clip_norm",
        ):
            payload.pop(key, None)
    else:
        payload["embedding_projection_enabled"] = True

    _log_trial_summary(
        logger,
        "Quick few-shot trial complete",
        final_composite_accuracy,
        after_train["accuracy"],
        after_val["accuracy"],
        after_heldout["accuracy"],
        elapsed_seconds,
    )
    return payload


def _average_metric_group(
    repeat_payloads: list[dict[str, Any]], phase: str, split: str
) -> dict[str, float]:
    metric_names = repeat_payloads[0][phase][split].keys()
    return {
        metric_name: float(
            np.mean(
                [
                    repeat_payload[phase][split][metric_name]
                    for repeat_payload in repeat_payloads
                ]
            )
        )
        for metric_name in metric_names
    }


def _metric_std(repeat_payloads: list[dict[str, Any]], metric_name: str) -> float:
    return float(
        np.std(
            [repeat_payload[metric_name] for repeat_payload in repeat_payloads],
            ddof=0,
        )
    )


def run_quick_trial(args: argparse.Namespace) -> dict[str, Any]:
    repeats = max(1, int(args.repeats))
    if repeats == 1:
        payload = _run_single_quick_trial(args)
        payload["repeats"] = 1
        payload["repeat_seed_stride"] = int(args.repeat_seed_stride)
        payload["repeat_metrics"] = [
            {
                "repeat": 1,
                "seed": int(payload["seed"]),
                "final_composite_accuracy": float(payload["final_composite_accuracy"]),
                "train_accuracy": float(payload["after"]["train"]["accuracy"]),
                "val_accuracy": float(payload["after"]["val"]["accuracy"]),
                "heldout_accuracy": float(payload["after"]["heldout"]["accuracy"]),
            }
        ]
        payload["metric_std"] = {
            "final_composite_accuracy": 0.0,
            "train_accuracy": 0.0,
            "val_accuracy": 0.0,
            "heldout_accuracy": 0.0,
        }
        return payload

    repeat_payloads = []
    start_time = time.perf_counter()
    base_seed = int(args.seed)
    seed_stride = int(args.repeat_seed_stride)
    for repeat_idx in range(repeats):
        repeat_args = argparse.Namespace(**vars(args))
        repeat_args.seed = base_seed + repeat_idx * seed_stride
        repeat_args.repeats = 1
        if args.model_architecture_output:
            repeat_args.model_architecture_output = _with_repeat_suffix(
                args.model_architecture_output,
                repeat_idx=repeat_idx,
                repeats=repeats,
            )
        repeat_payload = _run_single_quick_trial(repeat_args)
        repeat_payload["repeat"] = repeat_idx + 1
        repeat_payloads.append(repeat_payload)

    first_payload = repeat_payloads[0]
    averaged_payload = dict(first_payload)
    averaged_payload["elapsed_seconds"] = time.perf_counter() - start_time
    averaged_payload["seed"] = base_seed
    averaged_payload["repeats"] = repeats
    averaged_payload["repeat_seed_stride"] = seed_stride
    averaged_payload.pop("repeat", None)
    averaged_payload["update_history"] = []
    averaged_payload["repeat_seeds"] = [
        int(repeat_payload["seed"]) for repeat_payload in repeat_payloads
    ]
    averaged_payload["before"] = {
        split: _average_metric_group(repeat_payloads, "before", split)
        for split in ("train", "val", "heldout")
    }
    averaged_payload["after"] = {
        split: _average_metric_group(repeat_payloads, "after", split)
        for split in ("train", "val", "heldout")
    }
    averaged_payload["final_composite_accuracy"] = float(
        np.mean(
            [
                repeat_payload["final_composite_accuracy"]
                for repeat_payload in repeat_payloads
            ]
        )
    )
    averaged_payload["heldout_generalization_gap"] = float(
        averaged_payload["after"]["train"]["accuracy"]
        - averaged_payload["after"]["heldout"]["accuracy"]
    )
    averaged_payload["train_accuracy_delta"] = float(
        averaged_payload["after"]["train"]["accuracy"]
        - averaged_payload["before"]["train"]["accuracy"]
    )
    averaged_payload["val_accuracy_delta"] = float(
        averaged_payload["after"]["val"]["accuracy"]
        - averaged_payload["before"]["val"]["accuracy"]
    )
    averaged_payload["heldout_accuracy_delta"] = float(
        averaged_payload["after"]["heldout"]["accuracy"]
        - averaged_payload["before"]["heldout"]["accuracy"]
    )
    averaged_payload["repeat_metrics"] = [
        {
            "repeat": int(repeat_payload["repeat"]),
            "seed": int(repeat_payload["seed"]),
            "final_composite_accuracy": float(
                repeat_payload["final_composite_accuracy"]
            ),
            "train_accuracy": float(repeat_payload["after"]["train"]["accuracy"]),
            "val_accuracy": float(repeat_payload["after"]["val"]["accuracy"]),
            "heldout_accuracy": float(repeat_payload["after"]["heldout"]["accuracy"]),
        }
        for repeat_payload in repeat_payloads
    ]
    averaged_payload["metric_std"] = {
        "final_composite_accuracy": _metric_std(
            repeat_payloads, "final_composite_accuracy"
        ),
        "train_accuracy": float(
            np.std(
                [
                    repeat_payload["after"]["train"]["accuracy"]
                    for repeat_payload in repeat_payloads
                ],
                ddof=0,
            )
        ),
        "val_accuracy": float(
            np.std(
                [
                    repeat_payload["after"]["val"]["accuracy"]
                    for repeat_payload in repeat_payloads
                ],
                ddof=0,
            )
        ),
        "heldout_accuracy": float(
            np.std(
                [
                    repeat_payload["after"]["heldout"]["accuracy"]
                    for repeat_payload in repeat_payloads
                ],
                ddof=0,
            )
        ),
    }
    averaged_payload["repeat_payloads"] = repeat_payloads
    setup_logger("quick_fewshot_trial").info(
        "Averaged quick few-shot trial complete: "
        f"composite={averaged_payload['final_composite_accuracy']:.4f}, "
        f"train_acc={averaged_payload['after']['train']['accuracy']:.4f}, "
        f"val_acc={averaged_payload['after']['val']['accuracy']:.4f}, "
        f"heldout_acc={averaged_payload['after']['heldout']['accuracy']:.4f}, "
        f"repeats={repeats}, "
        f"elapsed_seconds={averaged_payload['elapsed_seconds']:.2f}"
    )
    return averaged_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a tiny LOSO few-shot probe for rapid architecture/training idea "
            "iteration without running the full overfit benchmark."
        )
    )
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--held-out-subject", type=int, default=None)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--task-batch-size", type=int, default=10)
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=1,
        help="Number of episodic tasks whose samples are encoded together.",
    )
    parser.add_argument("--train-eval-tasks", type=int, default=10)
    parser.add_argument("--val-tasks", type=int, default=10)
    parser.add_argument("--heldout-tasks", type=int, default=100)
    parser.add_argument("--k-shot", type=int, default=10)
    parser.add_argument("--q-query", type=int, default=10)
    parser.add_argument("--task-class-ids", type=str, default="0,4")
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
    parser.add_argument("--can-margin-loss-weight", type=float, default=0.2)
    parser.add_argument("--can-margin-target", type=float, default=0.3)
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
    parser.add_argument("--train-prefetch-batches", type=int, default=2)
    parser.add_argument(
        "--logging-verbosity",
        type=int,
        default=1,
        choices=(0, 1, 2),
        help="0=minimal, 1=standard, 2=detailed logging",
    )
    parser.add_argument(
        "--summary-every-n-updates",
        type=int,
        default=10,
        help="Emit train/val/heldout summary every N updates and at the end.",
    )
    parser.add_argument("--disable-window-shift", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--repeat-seed-stride", type=int, default=1)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument(
        "--model-architecture-output",
        type=str,
        default="./outputs/model_architecture/architecture.txt",
        help=(
            "Optional path to save a full model architecture summary after running "
            "one sample task through the model."
        ),
    )
    args = parser.parse_args()

    payload = run_quick_trial(args)
    if args.output_json:
        _write_json(Path(args.output_json), payload)
    if args.output_csv:
        _write_csv(Path(args.output_csv), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
