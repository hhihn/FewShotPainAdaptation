"""Visualize and summarize a completed PainNAS cross-fitted LOSO run.

The selected accuracy shown by this script is the inner subject-macro validation
accuracy used by NAS.  It is deliberately kept separate from the outer LOSO test
accuracy so that architecture selection and final evaluation are not conflated.

Example
-------
python -m painnas.visualize_cross_fitted_results \
    --run-dir data/cross_fitted_loso
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

BASE_RATE_METRICS = ("accuracy", "macro_f1", "auroc")
DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "precision_t4": "T4 precision",
    "recall_t4": "T4 recall",
    "precision_macro": "Macro precision",
    "recall_macro": "Macro recall",
    "auroc": "AUROC",
    "cross_entropy": "Cross-entropy",
}


def _is_multiclass(summary: dict[str, Any]) -> bool:
    """Return whether the aggregate confusion matrix represents more than two classes."""
    matrix = np.asarray(summary.get("aggregate_confusion_matrix", ()))
    return matrix.ndim == 2 and matrix.shape[0] > 2


def _rate_metrics(summary: dict[str, Any]) -> tuple[str, ...]:
    """Return metrics with labels appropriate for binary or multiclass output."""
    metrics = ["accuracy", "macro_f1"]
    metric_summary = summary.get("metrics", {})
    if _is_multiclass(summary) and {"precision_macro", "recall_macro"}.issubset(metric_summary):
        metrics.extend(("precision_macro", "recall_macro"))
    else:
        metrics.extend(("precision_t4", "recall_t4"))
    metrics.append("auroc")
    return tuple(metrics)


def _class_labels(summary: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    """Resolve display labels in the same order as the confusion-matrix axes."""
    matrix = np.asarray(summary["aggregate_confusion_matrix"])
    count = matrix.shape[0]
    class_ids = manifest.get("config", {}).get("raw_class_ids", [])
    if len(class_ids) == count:
        return [f"T{class_id}" for class_id in class_ids]
    return [f"Class {index}" for index in range(count)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load and minimally validate the analysis-ready run artifacts."""
    run_dir = Path(run_dir)
    metrics_path = run_dir / "fold_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Required result file does not exist: {metrics_path}")

    folds = pd.read_csv(metrics_path).sort_values("fold_index").reset_index(drop=True)
    required = {
        "fold_index",
        "outer_block_index",
        "selected_subject_accuracy_mean",
        "selected_subject_accuracy_standard_error",
        "selected_uncertainty_objective",
        *BASE_RATE_METRICS,
        "cross_entropy",
    }
    missing = sorted(required.difference(folds.columns))
    if missing:
        raise ValueError(f"{metrics_path} is missing columns: {', '.join(missing)}")
    if folds.empty:
        raise ValueError(f"No completed folds found in {metrics_path}")

    summary = _read_json(run_dir / "summary.json")
    manifest = _read_json(run_dir / "manifest.json")
    return folds, summary, manifest


def selected_block_metrics(folds: pd.DataFrame) -> pd.DataFrame:
    """Return one architecture-selection row per deterministic outer block."""
    columns = [
        "outer_block_index",
        "selected_trial",
        "selected_subject_accuracy_mean",
        "selected_subject_accuracy_standard_error",
        "selected_uncertainty_objective",
        "architecture_fingerprint",
    ]
    available = [column for column in columns if column in folds.columns]
    selected = folds[available].drop_duplicates().sort_values("outer_block_index")
    if selected["outer_block_index"].duplicated().any():
        raise ValueError("Selection metrics are inconsistent within an outer block")
    return selected.reset_index(drop=True)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _plot_fold_performance(ax: plt.Axes, folds: pd.DataFrame, summary: dict[str, Any]) -> None:
    x = folds["fold_index"].to_numpy()
    ax.plot(
        x,
        folds["accuracy"],
        color=OKABE_ITO["blue"],
        linewidth=1.1,
        marker="o",
        markersize=2.8,
        label="Test accuracy",
    )
    ax.plot(
        x,
        folds["macro_f1"],
        color=OKABE_ITO["orange"],
        linewidth=1.1,
        linestyle="--",
        marker="s",
        markersize=2.5,
        label="Test macro F1",
    )
    for metric, color in (("accuracy", OKABE_ITO["blue"]), ("macro_f1", OKABE_ITO["orange"])):
        mean = float(summary["metrics"][metric]["mean"])
        ax.axhline(mean, color=color, linewidth=1.0, alpha=0.75)
    ax.set(xlabel="LOSO fold (held-out subject)", ylabel="Score", ylim=(-0.02, 1.02))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.set_title("A  Held-out performance over folds", loc="left", fontweight="bold")
    _style_axes(ax)


def _plot_selected_accuracy(ax: plt.Axes, selected: pd.DataFrame, beta: float) -> None:
    blocks = selected["outer_block_index"].to_numpy(dtype=int)
    accuracy = selected["selected_subject_accuracy_mean"].to_numpy(dtype=float)
    standard_error = selected["selected_subject_accuracy_standard_error"].to_numpy(dtype=float)
    objective = selected["selected_uncertainty_objective"].to_numpy(dtype=float)
    ax.errorbar(
        blocks,
        accuracy,
        yerr=standard_error,
        fmt="o",
        color=OKABE_ITO["green"],
        ecolor=OKABE_ITO["green"],
        capsize=4,
        linewidth=1.4,
        label="Chosen validation accuracy ± SE",
    )
    ax.plot(
        blocks,
        objective,
        marker="D",
        markersize=4,
        linestyle="--",
        color=OKABE_ITO["purple"],
        label=rf"Selection objective (mean − {beta:g} × SE)",
    )
    for block, value in zip(blocks, accuracy):
        ax.annotate(f"{value:.1%}", (block, value+0.02), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7)
    ax.set(
        xlabel="Outer subject block",
        ylabel="Inner validation score",
        xticks=blocks,
        ylim=(max(0.0, float(np.min(objective - standard_error)) - 0.05), min(1.0, float(np.max(accuracy + standard_error)) + 0.07)),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("B  Accuracy used to choose each architecture", loc="left", fontweight="bold")
    _style_axes(ax)


def _plot_aggregate_metrics(ax: plt.Axes, summary: dict[str, Any]) -> None:
    metric_summary = summary["metrics"]
    present = [metric for metric in _rate_metrics(summary) if metric in metric_summary]
    means = np.asarray([metric_summary[metric]["mean"] for metric in present], dtype=float)
    low = np.asarray([metric_summary[metric]["ci_low"] for metric in present], dtype=float)
    high = np.asarray([metric_summary[metric]["ci_high"] for metric in present], dtype=float)
    positions = np.arange(len(present))
    ax.errorbar(
        means,
        positions,
        xerr=np.vstack((means - low, high - means)),
        fmt="o",
        color=OKABE_ITO["blue"],
        ecolor=OKABE_ITO["sky"],
        capsize=4,
        markersize=5,
    )
    for y, value in zip(positions, means):
        ax.annotate(f"{value:.1%}", (value+0.033, y), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set(
        xlabel="Aggregate score (95% bootstrap CI)",
        yticks=positions,
        yticklabels=[DISPLAY_NAMES[metric] for metric in present],
        xlim=(0.0, 1.02),
    )
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("C  Aggregate test metrics", loc="left", fontweight="bold")


def _plot_confusion_matrix(ax: plt.Axes, summary: dict[str, Any], manifest: dict[str, Any]) -> None:
    matrix = np.asarray(summary["aggregate_confusion_matrix"], dtype=int)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("aggregate_confusion_matrix must be a non-empty square matrix")
    labels = _class_labels(summary, manifest)
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    threshold = float(matrix.max()) / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontweight="bold",
            )
    ax.set(
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_title("D  Aggregate confusion matrix", loc="left", fontweight="bold")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Samples")


def create_figure(
    folds: pd.DataFrame,
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> plt.Figure:
    """Create a colorblind-safe overview figure for the run."""
    selected = selected_block_metrics(folds)
    beta = float(manifest.get("config", {}).get("uncertainty_beta", 1.0))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    _plot_fold_performance(axes[0, 0], folds, summary)
    _plot_selected_accuracy(axes[0, 1], selected, beta)
    _plot_aggregate_metrics(axes[1, 0], summary)
    _plot_confusion_matrix(axes[1, 1], summary, manifest)

    cross_entropy = summary["metrics"].get("cross_entropy", {}).get("mean")
    subtitle = (
        f"{len(folds)} completed folds | "
        f"{summary.get('unique_selected_architectures', selected.shape[0])} selected architectures"
    )
    if cross_entropy is not None:
        subtitle += f" | mean cross-entropy {float(cross_entropy):.3f}"
    return figure


def format_report(
    folds: pd.DataFrame,
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    """Build a concise, terminal-friendly report of the plotted values."""
    selected = selected_block_metrics(folds)
    beta = float(manifest.get("config", {}).get("uncertainty_beta", 1.0))
    lines = [
        "PainNAS cross-fitted LOSO run",
        f"Completed folds: {len(folds)}/{summary.get('total_folds', '?')}",
        "",
        "Architecture-selection accuracy (inner development subjects; not test accuracy):",
    ]
    for row in selected.itertuples(index=False):
        trial = f", trial {int(row.selected_trial)}" if hasattr(row, "selected_trial") else ""
        lines.append(
            f"  Block {int(row.outer_block_index)}{trial}: "
            f"{row.selected_subject_accuracy_mean:.3f} ± {row.selected_subject_accuracy_standard_error:.3f} SE; "
            f"objective={row.selected_uncertainty_objective:.3f}"
        )
    lines.extend([f"  Objective: mean accuracy - {beta:g} × SE", "", "Outer LOSO test metrics:"])
    for metric in (*_rate_metrics(summary), "cross_entropy"):
        values = summary.get("metrics", {}).get(metric)
        if not values:
            continue
        lines.append(
            f"  {DISPLAY_NAMES[metric]}: {values['mean']:.3f} "
            f"(95% CI {values['ci_low']:.3f}–{values['ci_high']:.3f}; fold SD {values['std']:.3f})"
        )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("../data/PainNAS/cross_fitted_run_late_mc"),
        help="Directory containing fold_metrics.csv, summary.json, and manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path (default: RUN_DIR/results_overview.png)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution (default: 300)")
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also save a vector PDF next to the PNG",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    output = (args.output or run_dir / "results_overview.png").expanduser().resolve()
    if args.dpi <= 0:
        raise ValueError("--dpi must be greater than zero")

    folds, summary, manifest = load_results(run_dir)
    figure = create_figure(folds, summary, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    saved = [output]
    pdf_path = output.with_suffix(".pdf")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    saved.append(pdf_path)
    plt.close(figure)

    print(format_report(folds, summary, manifest))
    print("\nSaved:")
    for path in saved:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
