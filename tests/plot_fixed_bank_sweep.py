import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logger import setup_logger


def _read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(row: dict, key: str) -> float:
    return float(row[key])


def _to_int(row: dict, key: str) -> int:
    return int(row[key])


def _series_key(row: dict) -> tuple[str, str, str]:
    return (
        row["fusion_method"],
        row["classifier_mode"],
        row["normalize_mode"],
    )


def _label(key: tuple[str, str, str]) -> str:
    fusion_method, classifier_mode, normalize_mode = key
    return f"{fusion_method} + {classifier_mode} ({normalize_mode})"


def plot_sweep(csv_path: Path, output_path: Path) -> None:
    rows = _read_rows(csv_path)
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_series_key(row)].append(row)

    for key in grouped:
        grouped[key].sort(key=lambda row: _to_int(row, "num_fixed_tasks"))

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metric_layout = [
        ("best_accuracy", "Best Accuracy", axes[0][0]),
        ("last_accuracy", "Final Accuracy", axes[0][1]),
        ("best_loss", "Best Loss", axes[1][0]),
        ("last_loss", "Final Loss", axes[1][1]),
    ]

    for key, rows_for_key in grouped.items():
        x = [_to_int(row, "num_fixed_tasks") for row in rows_for_key]
        for metric_name, title, axis in metric_layout:
            y = [_to_float(row, metric_name) for row in rows_for_key]
            axis.plot(x, y, marker="o", linewidth=2, label=_label(key))
            axis.set_title(title)
            axis.set_xlabel("Fixed Task Bank Size")
            axis.grid(True, alpha=0.3)

    axes[0][0].set_ylabel("Accuracy")
    axes[1][0].set_ylabel("Loss")

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("Fixed-Bank Overfit Sweep", fontsize=14)
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the aggregated fixed-bank overfit sweep CSV."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="tests/outputs/diagnostics/fixed_bank_overfit_sweep.csv",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="tests/outputs/diagnostics/fixed_bank_overfit_sweep.png",
    )
    args = parser.parse_args()

    logger = setup_logger("plot_fixed_bank_sweep")
    csv_path = Path(args.csv_path)
    output_path = Path(args.output_path)
    plot_sweep(csv_path=csv_path, output_path=output_path)
    logger.info(f"Saved sweep plot to {output_path}")


if __name__ == "__main__":
    main()
