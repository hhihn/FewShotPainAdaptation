import argparse
import csv
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import sys


# Allow running this file directly from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.overfit_single_episode import run_fixed_episode_bank_overfit
from utils.logger import setup_logger


@dataclass
class SweepResult:
    fusion_method: str
    classifier_mode: str
    normalize_mode: str
    num_fixed_tasks: int
    seed: int
    num_batches: int
    learning_rate: float
    first_loss: float
    best_loss: float
    last_loss: float
    first_accuracy: float
    best_accuracy: float
    last_accuracy: float
    best_accuracy_step: int
    first_reach_95_step: int | None
    passed: bool


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _first_reach_step(history, threshold: float) -> int | None:
    for metrics in history:
        if metrics.accuracy >= threshold:
            return metrics.step
    return None


def _format_step(step: int | None) -> str:
    return "-" if step is None else str(step)


def _write_results_csv(path: Path, results: list[SweepResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed-bank overfit sweep across fusion, classifier, "
            "normalization, and task-bank size settings."
        )
    )
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-batches", type=int, default=1000)
    parser.add_argument("--task-batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--k-shot", type=int, default=3)
    parser.add_argument("--q-query", type=int, default=3)
    parser.add_argument(
        "--fusion-methods",
        type=str,
        default="mean,gated",
        help="Comma-separated fusion modes to evaluate.",
    )
    parser.add_argument(
        "--classifier-modes",
        type=str,
        default="prototype,soft_knn",
        help="Comma-separated classifier modes to evaluate.",
    )
    parser.add_argument(
        "--normalize-modes",
        type=str,
        default="subject",
        help="Comma-separated normalization modes to evaluate.",
    )
    parser.add_argument(
        "--task-sizes",
        type=str,
        default="1,3,5,10,20",
        help="Comma-separated fixed task-bank sizes to evaluate.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--results-csv",
        type=str,
        default="outputs/diagnostics/fixed_bank_overfit_sweep.csv",
        help="Optional CSV path for aggregated sweep results.",
    )
    args = parser.parse_args()

    logger = setup_logger("fixed_bank_overfit_sweep")
    fusion_methods = _parse_csv_list(args.fusion_methods)
    classifier_modes = _parse_csv_list(args.classifier_modes)
    normalize_modes = _parse_csv_list(args.normalize_modes)
    task_sizes = _parse_int_list(args.task_sizes)

    combinations = list(
        product(fusion_methods, classifier_modes, normalize_modes, task_sizes)
    )
    logger.info(
        f"Starting fixed-bank overfit sweep with {len(combinations)} runs: "
        f"fusion_methods={fusion_methods}, "
        f"classifier_modes={classifier_modes}, "
        f"normalize_modes={normalize_modes}, "
        f"task_sizes={task_sizes}"
    )

    results: list[SweepResult] = []
    for run_index, (fusion_method, classifier_mode, normalize_mode, num_fixed_tasks) in enumerate(
        combinations, start=1
    ):
        logger.info(
            f"[Run {run_index}/{len(combinations)}] "
            f"fusion={fusion_method}, classifier={classifier_mode}, "
            f"normalize={normalize_mode}, tasks={num_fixed_tasks}"
        )
        history, passed = run_fixed_episode_bank_overfit(
            data_dir=args.data_dir,
            seed=args.seed,
            fusion_method=fusion_method,
            num_batches=args.num_batches,
            task_batch_size=args.task_batch_size,
            learning_rate=args.learning_rate,
            k_shot=args.k_shot,
            q_query=args.q_query,
            log_every=args.log_every,
            num_fixed_tasks=num_fixed_tasks,
            normalize_mode=normalize_mode,
            classifier_mode=classifier_mode,
        )

        first = history[0]
        last = history[-1]
        best_loss_metrics = min(history, key=lambda item: item.loss)
        best_acc_metrics = max(history, key=lambda item: item.accuracy)
        results.append(
            SweepResult(
                fusion_method=fusion_method,
                classifier_mode=classifier_mode,
                normalize_mode=normalize_mode,
                num_fixed_tasks=num_fixed_tasks,
                seed=args.seed,
                num_batches=args.num_batches,
                learning_rate=args.learning_rate,
                first_loss=first.loss,
                best_loss=best_loss_metrics.loss,
                last_loss=last.loss,
                first_accuracy=first.accuracy,
                best_accuracy=best_acc_metrics.accuracy,
                last_accuracy=last.accuracy,
                best_accuracy_step=best_acc_metrics.step,
                first_reach_95_step=_first_reach_step(history, threshold=0.95),
                passed=passed,
            )
        )

    if results:
        csv_path = Path(args.results_csv)
        _write_results_csv(csv_path, results)
        logger.info(f"Saved sweep results to {csv_path}")

        header = (
            "fusion      classifier normalize tasks best_acc last_acc "
            "best_loss last_loss step@best step@0.95 pass"
        )
        logger.info(header)
        logger.info("-" * len(header))
        for result in results:
            logger.info(
                f"{result.fusion_method:<11} "
                f"{result.classifier_mode:<10} "
                f"{result.normalize_mode:<9} "
                f"{result.num_fixed_tasks:<5} "
                f"{result.best_accuracy:<8.4f} "
                f"{result.last_accuracy:<8.4f} "
                f"{result.best_loss:<9.4f} "
                f"{result.last_loss:<9.4f} "
                f"{result.best_accuracy_step:<9} "
                f"{_format_step(result.first_reach_95_step):<9} "
                f"{str(result.passed)}"
            )


if __name__ == "__main__":
    main()
