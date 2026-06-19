#!/usr/bin/env python3
"""Aggregate the primary matched-query personalization experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def holm_adjust(values: np.ndarray) -> np.ndarray:
    """Return Holm family-wise adjusted p-values."""
    pvalues = np.asarray(values, dtype=np.float64)
    adjusted = np.full(pvalues.shape, np.nan, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(pvalues))
    order = finite[np.argsort(pvalues[finite])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    trials: int = 20_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI with subjects as the resampling unit."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(int(trials), dtype=np.float64)
    for start in range(0, int(trials), 1_000):
        count = min(1_000, int(trials) - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = np.mean(values[indices], axis=1)
    alpha = 1.0 - float(confidence)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def resolve_repeat_files(inputs: Iterable[str | Path]) -> list[Path]:
    """Resolve explicit files and directories to repeat-metric CSVs."""
    paths: list[Path] = []
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(path.rglob("*_matched_query_repeats.csv")))
        else:
            raise FileNotFoundError(path)
    unique = sorted({path.resolve() for path in paths})
    if not unique:
        raise FileNotFoundError("No *_matched_query_repeats.csv files found")
    return unique


def load_repeat_metrics(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and validate per-fold matched-query repeat rows."""
    frames = [pd.read_csv(path) for path in resolve_repeat_files(paths)]
    data = pd.concat(frames, ignore_index=True)
    required = {
        "test_subject",
        "repeat",
        "query_hash",
        "zero_shot_accuracy",
        "k_shot_accuracy",
        "difference_accuracy",
        "zero_shot_macro_f1",
        "k_shot_macro_f1",
        "difference_macro_f1",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Matched-query rows are missing columns: {sorted(missing)}")
    duplicate = data.duplicated(["test_subject", "repeat"], keep=False)
    if duplicate.any():
        raise ValueError("Duplicate subject/repeat rows in matched-query results")
    query_hash_counts = data.groupby("test_subject")["query_hash"].nunique()
    if (query_hash_counts != 1).any():
        subjects = query_hash_counts[query_hash_counts != 1].index.tolist()
        raise ValueError(f"Query content changed across repeats for subjects={subjects}")
    repeat_counts = data.groupby("test_subject")["repeat"].nunique()
    if repeat_counts.nunique() != 1:
        raise ValueError(
            "Subjects have unequal support-repeat counts: "
            + repeat_counts.to_dict().__repr__()
        )
    expected_repeats = np.arange(int(repeat_counts.iloc[0]))
    for subject, rows in data.groupby("test_subject"):
        observed = np.sort(rows["repeat"].to_numpy(dtype=np.int64))
        if not np.array_equal(observed, expected_repeats):
            raise ValueError(
                f"Support repeats are incomplete for subject={subject}: "
                f"observed={observed.tolist()}"
            )
    return data.sort_values(["test_subject", "repeat"]).reset_index(drop=True)


def metric_specs(data: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    """Return metric name, role, zero column, and k-shot column."""
    specs = [
        ("macro_f1", "primary", "zero_shot_macro_f1", "k_shot_macro_f1"),
        ("accuracy", "secondary", "zero_shot_accuracy", "k_shot_accuracy"),
    ]
    class_indices = sorted(
        int(column.rsplit("_", 1)[1])
        for column in data.columns
        if column.startswith("zero_shot_f1_class_")
    )
    for class_index in class_indices:
        specs.append(
            (
                f"f1_class_{class_index}",
                "exploratory",
                f"zero_shot_f1_class_{class_index}",
                f"k_shot_f1_class_{class_index}",
            )
        )
    return specs


def build_subject_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Average support repeats within subject and retain sampling variance."""
    records = []
    for metric, role, zero_column, k_column in metric_specs(data):
        for subject, rows in data.groupby("test_subject", sort=True):
            zero_values = rows[zero_column].to_numpy(dtype=np.float64)
            if not np.allclose(zero_values, zero_values[0], rtol=0.0, atol=0.0):
                raise ValueError(
                    f"Zero-shot {metric} changed across support repeats for subject={subject}"
                )
            k_values = rows[k_column].to_numpy(dtype=np.float64)
            differences = k_values - zero_values[0]
            records.append(
                {
                    "test_subject": int(subject),
                    "metric": metric,
                    "endpoint_role": role,
                    "support_repeats": int(len(rows)),
                    "zero_shot": float(zero_values[0]),
                    "k_shot_mean": float(np.mean(k_values)),
                    "mean_difference": float(np.mean(differences)),
                    "support_draw_sd": (
                        float(np.std(k_values, ddof=1)) if len(k_values) > 1 else math.nan
                    ),
                    "support_draw_variance": (
                        float(np.var(k_values, ddof=1)) if len(k_values) > 1 else math.nan
                    ),
                    "support_draw_q025": float(np.quantile(k_values, 0.025)),
                    "support_draw_q975": float(np.quantile(k_values, 0.975)),
                }
            )
    return pd.DataFrame(records).sort_values(["metric", "test_subject"])


def paired_statistics(
    subject_metrics: pd.DataFrame,
    *,
    bootstrap_trials: int = 20_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute subject-level paired inference for every requested endpoint."""
    records = []
    for metric_index, (metric, rows) in enumerate(
        subject_metrics.groupby("metric", sort=False)
    ):
        zero = rows["zero_shot"].to_numpy(dtype=np.float64)
        k_shot = rows["k_shot_mean"].to_numpy(dtype=np.float64)
        difference = k_shot - zero
        if len(rows) > 1:
            t_result = stats.ttest_rel(k_shot, zero)
            t_statistic = float(t_result.statistic)
            t_pvalue = float(t_result.pvalue)
        else:
            t_statistic = math.nan
            t_pvalue = math.nan
        if np.allclose(difference, 0.0):
            wilcoxon_statistic, wilcoxon_pvalue = 0.0, 1.0
        else:
            wilcoxon_result = stats.wilcoxon(
                k_shot, zero, zero_method="wilcox", alternative="two-sided"
            )
            wilcoxon_statistic = float(wilcoxon_result.statistic)
            wilcoxon_pvalue = float(wilcoxon_result.pvalue)
        difference_sd = (
            float(np.std(difference, ddof=1)) if difference.size > 1 else math.nan
        )
        ci_low, ci_high = bootstrap_mean_ci(
            difference,
            trials=bootstrap_trials,
            confidence=confidence,
            seed=seed + metric_index,
        )
        records.append(
            {
                "metric": metric,
                "endpoint_role": rows["endpoint_role"].iloc[0],
                "n_subjects": int(len(rows)),
                "zero_shot_mean": float(np.mean(zero)),
                "k_shot_mean": float(np.mean(k_shot)),
                "mean_difference": float(np.mean(difference)),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "paired_t_statistic": t_statistic,
                "degrees_freedom": int(len(rows) - 1),
                "paired_t_pvalue": t_pvalue,
                "cohen_dz": (
                    float(np.mean(difference) / difference_sd)
                    if np.isfinite(difference_sd) and difference_sd > 0
                    else math.nan
                ),
                "wilcoxon_statistic": wilcoxon_statistic,
                "wilcoxon_pvalue": wilcoxon_pvalue,
                "mean_support_draw_sd": float(rows["support_draw_sd"].mean()),
                "median_support_draw_sd": float(rows["support_draw_sd"].median()),
            }
        )
    result = pd.DataFrame(records)
    result["paired_t_pvalue_holm"] = np.nan
    result["wilcoxon_pvalue_holm"] = np.nan
    class_mask = result["metric"].str.startswith("f1_class_")
    result.loc[class_mask, "paired_t_pvalue_holm"] = holm_adjust(
        result.loc[class_mask, "paired_t_pvalue"].to_numpy()
    )
    result.loc[class_mask, "wilcoxon_pvalue_holm"] = holm_adjust(
        result.loc[class_mask, "wilcoxon_pvalue"].to_numpy()
    )
    role_order = {"primary": 0, "secondary": 1, "exploratory": 2}
    result["_role_order"] = result["endpoint_role"].map(role_order)
    return result.sort_values(["_role_order", "metric"]).drop(
        columns="_role_order"
    ).reset_index(drop=True)


def run_analysis(
    inputs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    bootstrap_trials: int = 20_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """Run aggregation, write machine-readable outputs, and return a summary."""
    data = load_repeat_metrics(inputs)
    subjects = build_subject_metrics(data)
    inference = paired_statistics(
        subjects,
        bootstrap_trials=bootstrap_trials,
        confidence=confidence,
        seed=seed,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    subject_path = output / "matched_query_subject_metrics.csv"
    stats_path = output / "matched_query_paired_statistics.csv"
    uncertainty_path = output / "matched_query_support_uncertainty.csv"
    subjects.to_csv(subject_path, index=False)
    inference.to_csv(stats_path, index=False)
    subjects[
        [
            "test_subject",
            "metric",
            "support_repeats",
            "support_draw_sd",
            "support_draw_variance",
            "support_draw_q025",
            "support_draw_q975",
        ]
    ].to_csv(uncertainty_path, index=False)

    primary = inference[inference["metric"] == "macro_f1"].iloc[0].to_dict()
    summary = {
        "analysis_name": "matched_query_primary_personalization_test",
        "primary_endpoint": "macro_f1",
        "secondary_endpoint": "accuracy",
        "support_repeats_per_subject": int(data.groupby("test_subject").size().min()),
        "n_subjects": int(data["test_subject"].nunique()),
        "complete_biovid_loso": int(data["test_subject"].nunique()) == 87,
        "bootstrap_trials": int(bootstrap_trials),
        "confidence": float(confidence),
        "primary_result": primary,
        "subject_metrics_file": str(subject_path),
        "paired_statistics_file": str(stats_path),
        "support_uncertainty_file": str(uncertainty_path),
    }
    json_path = output / "matched_query_primary_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=float) + "\n")
    markdown_path = output / "matched_query_primary_summary.md"
    markdown_path.write_text(
        "# Matched-query primary personalisation test\n\n"
        "The inferential unit is the held-out subject. Macro-F1 is the primary "
        "endpoint; accuracy is secondary and per-class F1 is exploratory.\n\n"
        f"- Subjects: {summary['n_subjects']}\n"
        f"- Support draws per subject: {summary['support_repeats_per_subject']}\n"
        f"- Mean macro-F1 difference: {primary['mean_difference']:.6f}\n"
        f"- 95% bootstrap CI: [{primary['bootstrap_ci_low']:.6f}, "
        f"{primary['bootstrap_ci_high']:.6f}]\n"
        f"- Paired t({int(primary['degrees_freedom'])}) = "
        f"{primary['paired_t_statistic']:.6f}, p = {primary['paired_t_pvalue']:.12g}\n"
        f"- Cohen's dz: {primary['cohen_dz']:.6f}\n",
        encoding="utf-8",
    )
    summary["summary_json_file"] = str(json_path)
    summary["summary_markdown_file"] = str(markdown_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/matched_query_personalization"),
    )
    parser.add_argument("--bootstrap-trials", type=int, default=20_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = run_analysis(
        args.inputs,
        args.output_dir,
        bootstrap_trials=args.bootstrap_trials,
        confidence=args.confidence,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
