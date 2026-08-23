"""Print the results of a finished LOSO run.

``full_loso_payload.json`` holds everything, but at ~170 KB it is not something
you can read in Drive. This prints the headline metrics, the run's window and
sensor configuration, and the per-fold table.

Usage
-----
    python scripts/show_results.py                       # newest run under the run root
    python scripts/show_results.py <run_dir>
    python scripts/show_results.py <run_dir> --folds     # add the per-fold table
    python scripts/show_results.py <run_dir> --csv       # also write summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_RUN_ROOT = "/content/drive/MyDrive/FewShotPainAdaptationRuns"

HEADLINE = (
    ("zero_shot_accuracy", "zero-shot accuracy"),
    ("k_shot_accuracy", "k-shot accuracy"),
    ("source_subject_prototype_vote_accuracy", "prototype-vote accuracy"),
    ("zero_shot_f1", "zero-shot macro F1"),
    ("k_shot_f1", "k-shot macro F1"),
    ("zero_shot_loss", "zero-shot loss"),
    ("k_shot_loss", "k-shot loss"),
)


def resolve_payload(target: str | None) -> Path:
    """Return the payload path for an explicit target or the newest run."""
    if target:
        path = Path(target)
        if path.is_dir():
            path = path / "full_loso_payload.json"
        if not path.exists():
            raise SystemExit(f"No payload at {path}")
        return path

    root = Path(DEFAULT_RUN_ROOT)
    if not root.is_dir():
        raise SystemExit(
            f"No run directory given and {root} does not exist. "
            "Pass the run directory explicitly."
        )
    payloads = sorted(
        root.glob("*/full_loso_payload.json"), key=lambda p: p.stat().st_mtime
    )
    if not payloads:
        raise SystemExit(f"No finished runs under {root}")
    return payloads[-1]


def per_fold_columns(cv_results: dict, num_folds: int) -> dict[str, list]:
    """Collect every per-fold numeric series, keyed by metric name."""
    columns = {}
    for key, value in sorted(cv_results.items()):
        if not isinstance(value, list) or len(value) != num_folds:
            continue
        if not all(isinstance(item, (int, float)) for item in value):
            continue
        columns[key] = value
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", nargs="?", default=None,
                        help="Run directory or payload path (default: newest run)")
    parser.add_argument("--folds", action="store_true", help="Print the per-fold table")
    parser.add_argument("--csv", action="store_true",
                        help="Write summary.csv and fold_metrics.csv beside the payload")
    args = parser.parse_args()

    payload_path = resolve_payload(args.run)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    config = payload.get("config", {})
    cv_results = payload.get("cv_results", {})
    num_folds = int(summary.get("num_folds", 0))

    print(f"run          : {payload_path.parent.name}")
    print(f"folds        : {num_folds}"
          f"  (indices {config.get('loso_start_index')}..{config.get('loso_stop_index')})")
    print(f"elapsed      : {payload.get('elapsed_seconds', 0) / 3600:.2f} h")
    print(f"encoder      : {config.get('encoder_backend')}")
    print(f"channels     : {config.get('sensor_idx')} {config.get('modality_names')}")
    print(f"classes      : {config.get('task_class_ids')}")

    if config.get("window_shift_enabled"):
        width = config.get("window_shift_window_seconds")
        low = config.get("window_shift_start_min_seconds")
        high = config.get("window_shift_start_max_seconds")
        eval_start = config.get("window_shift_eval_start_seconds")
        eval_start = low if eval_start is None else eval_start
        print(f"window       : {width:g}s, train starts {low:g}-{high:g}s, "
              f"eval {eval_start:g}-{eval_start + width:g}s")
    else:
        print("window       : OFF (full signal)")

    print()
    print(f"  {'metric':<26} {'mean':>9} {'std':>9}")
    print(f"  {'-' * 26} {'-' * 9} {'-' * 9}")
    for key, label in HEADLINE:
        stats = summary.get(key)
        if stats:
            print(f"  {label:<26} {stats['mean']:>9.4f} {stats['std']:>9.4f}")

    columns = per_fold_columns(cv_results, num_folds)
    if args.folds and columns:
        interesting = [
            key for key in (
                "zero_shot_accuracies", "k_shot_accuracies",
                "source_subject_prototype_vote_accuracies",
                "train_accuracies", "val_accuracies",
            ) if key in columns
        ]
        print()
        # Truncate: "source_subject_prototype_vote" overflows the column.
        header = "  fold  " + "".join(
            f"{k.replace('_accuracies', '')[:16]:>18}" for k in interesting
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for index in range(num_folds):
            row = "".join(f"{columns[k][index]:>18.4f}" for k in interesting)
            print(f"  {index + 1:>4}  {row}")

    if args.csv:
        summary_csv = payload_path.parent / "summary.csv"
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "mean", "std"])
            for key, stats in sorted(summary.items()):
                if isinstance(stats, dict):
                    writer.writerow([key, stats.get("mean"), stats.get("std")])
        folds_csv = payload_path.parent / "fold_metrics.csv"
        with folds_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["fold"] + list(columns))
            for index in range(num_folds):
                writer.writerow([index + 1] + [columns[k][index] for k in columns])
        print(f"\nwrote {summary_csv.name} and {folds_csv.name}")


if __name__ == "__main__":
    main()
