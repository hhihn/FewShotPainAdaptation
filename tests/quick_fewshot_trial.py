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
    loss, accuracy, contrastive_loss = learner.evaluate_batch_step(tasks)
    return {
        "loss": float(loss.numpy()),
        "accuracy": float(accuracy.numpy()),
        "contrastive_loss": float(contrastive_loss.numpy()),
    }


def _cyclic_task_batch(
    tasks: list[dict[str, np.ndarray]],
    update_idx: int,
    task_batch_size: int,
) -> list[dict[str, np.ndarray]]:
    start = update_idx * task_batch_size
    return [tasks[(start + offset) % len(tasks)] for offset in range(task_batch_size)]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_payload = {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in payload.items()
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_payload.keys()))
        writer.writeheader()
        writer.writerow(flat_payload)


def run_quick_trial(args: argparse.Namespace) -> dict[str, Any]:
    logger = setup_logger("quick_fewshot_trial")
    start_time = time.perf_counter()
    filters_list = _parse_int_tuple(args.filters)

    config = PainDatasetConfig(
        seed=args.seed,
        deterministic_ops=True,
        task_class_ids=_parse_int_tuple(args.task_class_ids),
        k_shot=args.k_shot,
        q_query=args.q_query,
        task_normalize_mode=args.normalize_mode,
        classifier_mode=args.classifier_mode,
        train_batch_size=args.task_batch_size,
        tasks_per_epoch=max(1, args.updates * args.task_batch_size),
        val_tasks=max(1, args.val_tasks),
        subject_eval_tasks=max(1, args.heldout_tasks),
        num_epochs=1,
        single_loso_fold=True,
        embedding_dim=args.embedding_dim,
        num_tcn_blocks=len(filters_list),
        filters_list=filters_list,
        tcn_attention_key_dim=args.tcn_attention_key_dim,
        tcn_attention_pool_size=args.tcn_attention_pool_size,
        enable_window_shift_augmentation=not args.disable_window_shift,
        gaussian_noise_std=args.gaussian_noise_std,
    )
    learner = FewShotPainLearner(
        config=config,
        data_dir=args.data_dir,
        learning_rate=args.learning_rate,
        fusion_method=args.fusion_method,
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
    train_eval_tasks = _sample_subject_tasks(fold["train_sampler"], args.train_eval_tasks)
    val_tasks = _sample_subject_tasks(fold["val_sampler"], args.val_tasks)
    heldout_tasks = _sample_tasks(fold["test_sampler"], args.heldout_tasks)

    before_train = _evaluate_bank(learner, train_eval_tasks)
    before_val = _evaluate_bank(learner, val_tasks)
    before_heldout = _evaluate_bank(learner, heldout_tasks)

    update_history = []
    for update_idx in range(max(1, args.updates)):
        task_batch = _cyclic_task_batch(train_tasks, update_idx, args.task_batch_size)
        loss, task_loss, accuracy, contrastive_loss = learner.train_batch_step(task_batch)
        update_history.append(
            {
                "update": update_idx + 1,
                "loss": float(loss.numpy()),
                "task_loss": float(task_loss.numpy()),
                "accuracy": float(accuracy.numpy()),
                "contrastive_loss": float(contrastive_loss.numpy()),
            }
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
    heldout_generalization_gap = float(after_train["accuracy"] - after_heldout["accuracy"])

    payload: dict[str, Any] = {
        "script": "tests/quick_fewshot_trial.py",
        "elapsed_seconds": elapsed_seconds,
        "seed": args.seed,
        "held_out_subject": held_out_subject,
        "train_subject_count": int(fold["n_train_subjects"]),
        "val_subject_count": int(fold["n_val_subjects"]),
        "updates": max(1, args.updates),
        "task_batch_size": args.task_batch_size,
        "train_task_count": len(train_tasks),
        "train_eval_task_count": len(train_eval_tasks),
        "val_task_count": len(val_tasks),
        "heldout_task_count": len(heldout_tasks),
        "k_shot": args.k_shot,
        "q_query": args.q_query,
        "task_class_ids": list(config.task_class_ids),
        "fusion_method": args.fusion_method,
        "classifier_mode": args.classifier_mode,
        "normalize_mode": args.normalize_mode,
        "embedding_dim": args.embedding_dim,
        "filters": list(filters_list),
        "window_shift_enabled": bool(config.enable_window_shift_augmentation),
        "gaussian_noise_std": float(config.gaussian_noise_std),
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
        "train_accuracy_delta": float(after_train["accuracy"] - before_train["accuracy"]),
        "val_accuracy_delta": float(after_val["accuracy"] - before_val["accuracy"]),
        "heldout_accuracy_delta": float(
            after_heldout["accuracy"] - before_heldout["accuracy"]
        ),
    }

    logger.info(
        "Quick few-shot trial complete: "
        f"composite={final_composite_accuracy:.4f}, "
        f"train_acc={after_train['accuracy']:.4f}, "
        f"val_acc={after_val['accuracy']:.4f}, "
        f"heldout_acc={after_heldout['accuracy']:.4f}, "
        f"elapsed_seconds={elapsed_seconds:.2f}"
    )
    return payload


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
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--task-batch-size", type=int, default=2)
    parser.add_argument("--train-eval-tasks", type=int, default=50)
    parser.add_argument("--val-tasks", type=int, default=10)
    parser.add_argument("--heldout-tasks", type=int, default=10)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--q-query", type=int, default=5)
    parser.add_argument("--task-class-ids", type=str, default="0,5")
    parser.add_argument(
        "--fusion-method",
        type=str,
        default="gated",
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
        default="split",
        choices=("subject", "split", "support", "none"),
    )
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--filters", type=str, default="8,16")
    parser.add_argument("--tcn-attention-key-dim", type=int, default=32)
    parser.add_argument("--tcn-attention-pool-size", type=int, default=4)
    parser.add_argument("--gaussian-noise-std", type=float, default=0.0)
    parser.add_argument("--disable-window-shift", action="store_true")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--output-csv", type=str, default="")
    args = parser.parse_args()

    payload = run_quick_trial(args)
    if args.output_json:
        _write_json(Path(args.output_json), payload)
    if args.output_csv:
        _write_csv(Path(args.output_csv), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
