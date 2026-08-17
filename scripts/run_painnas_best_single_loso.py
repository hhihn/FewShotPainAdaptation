#!/usr/bin/env python3
"""Train the block-1 PainNAS winner from scratch for one LOSO target.

This is the fixed-architecture (non-NAS, non-warm-started) LOSO experiment.  A
single invocation selects exactly one held-out BioVid subject, estimates
normalization on the other subjects' training samples, uses only source-subject
validation samples for early stopping, and evaluates the target once.

By default the architecture is loaded from the cross-fitted block-1 winner:
``data/cross_fitted_loso_new/blocks/block_001/search/best_architecture.json``.
That search excluded the block-1 subjects recorded in the JSON.  Testing a
different target has direct selection leakage and requires
``--allow-selection-leakage``.  Because choosing block 1 as the best of all
five winners is itself a post-search comparison, all resulting estimates
should still be described as exploratory unless that choice was pre-specified.

Example
-------
./venv/bin/python scripts/run_painnas_best_single_loso.py \
    --data-dir "data/BioVid 2" \
    --target-fold 1 \
    --batch-size 128 \
    --max-epochs 100 \
    --patience 15
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fewshotpain-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/fewshotpain-cache")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from painnas.config import PainNASConfig
from painnas.data import build_loso_fold_indices, load_biovid_binary
from painnas.io import read_json, to_jsonable
from painnas.loso import run_loso
from painnas.model import ArchitectureSpec, build_early_fusion_model
from painnas.runtime import require_gpu, reset_runtime
from painnas.search import load_architecture

DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "painnas" / "block_001_best_single_loso"


def _integer_tuple(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive comma-separated integers")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="BioVid dataset root")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory; a target-specific directory is created below it",
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--target-fold",
        type=int,
        help="One-based LOSO fold in sorted subject order (1 through 87 for BioVid)",
    )
    target.add_argument(
        "--target-subject",
        type=int,
        help="Internal integer subject ID stored by the dataset",
    )
    target.add_argument(
        "--target-key",
        help="Original predefined BioVid subject key, for example 120713-09_w_21",
    )

    training = parser.add_argument_group("training")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--batch-size", type=int, default=128)
    training.add_argument("--max-epochs", type=int, default=100)
    training.add_argument("--patience", type=int, default=15)
    training.add_argument("--bootstrap-samples", type=int, default=10_000)
    training.add_argument("--max-parameters", type=int, default=25_000_000)
    training.add_argument("--resume", action="store_true")
    training.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution (intended for smoke tests)",
    )
    training.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    training.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data, resolve the split, and build the model without fitting",
    )
    training.add_argument(
        "--allow-selection-leakage",
        action="store_true",
        help=(
            "Permit a target that was used by the block-1 architecture search. "
            "Results for such a target are exploratory, not selection-independent."
        ),
    )

    architecture = parser.add_argument_group(
        "optional architecture overrides",
        "Omitted values come directly from the selected block-1 JSON.",
    )
    architecture.add_argument("--num-blocks", type=int, choices=(3, 4, 5))
    architecture.add_argument(
        "--conv-repeats",
        type=_integer_tuple,
        help="Comma-separated repeats, one value per block (each must be 1 or 2)",
    )
    architecture.add_argument("--width-multiplier", type=float, choices=(0.5, 1.0, 2.0))
    architecture.add_argument("--temporal-kernel-size", type=int, choices=(7, 11, 15))
    architecture.add_argument(
        "--dense-units",
        type=_integer_tuple,
        help="One or two comma-separated hidden widths, e.g. 512 or 1024,512",
    )
    architecture.add_argument("--learning-rate", type=float)
    architecture.add_argument("--head-type", choices=("flatten", "global_average"))
    architecture.add_argument("--convolution-type", choices=("standard", "separable"))
    architecture.add_argument("--normalization-type", choices=("batch", "group", "layer"))
    architecture.add_argument("--pooling-type", choices=("max", "average"))
    architecture.add_argument("--pooling-size", type=int, choices=(2, 4))
    return parser


def architecture_from_args(
     args: argparse.Namespace
) -> ArchitectureSpec:
    """Apply explicitly supplied architecture overrides to the selected winner."""
    payload = {
        "num_blocks": args.num_blocks,
        "conv_repeats": args.conv_repeats,
        "width_multiplier": args.width_multiplier,
        "temporal_kernel_size": args.temporal_kernel_size,
        "dense_units": args.dense_units,
        "learning_rate": args.learning_rate,
        "head_type": args.head_type,
        "convolution_type": args.convolution_type,
        "normalization_type": args.normalization_type,
        "pooling_type": args.pooling_type,
        "pooling_size": args.pooling_size,
    }
    if len(payload["conv_repeats"]) != int(payload["num_blocks"]):
        raise ValueError(
            "--conv-repeats must contain exactly one value per block; provide it "
            "when changing --num-blocks"
        )
    return ArchitectureSpec.from_dict(payload)


def resolve_target(arrays: Any, args: argparse.Namespace) -> tuple[int, int, str]:
    """Return internal subject ID, one-based fold index, and display key."""
    ordered = [int(value) for value in sorted(arrays.unique_subjects.tolist())]
    if args.target_fold is not None:
        if not 1 <= args.target_fold <= len(ordered):
            raise ValueError(f"--target-fold must be between 1 and {len(ordered)}")
        fold_index = int(args.target_fold)
        subject = ordered[fold_index - 1]
    elif args.target_subject is not None:
        subject = int(args.target_subject)
        if subject not in ordered:
            raise ValueError(f"Unknown --target-subject {subject}; available IDs: {ordered}")
        fold_index = ordered.index(subject) + 1
    else:
        matches = [
            subject
            for subject, key in arrays.subject_keys.items()
            if key == str(args.target_key)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"--target-key must match exactly one subject; found {len(matches)} "
                f"matches for {args.target_key!r}"
            )
        subject = int(matches[0])
        fold_index = ordered.index(subject) + 1
    return subject, fold_index, arrays.subject_keys.get(subject, str(subject))


def _winner_outer_subjects(architecture_path: Path) -> set[int]:
    payload = read_json(architecture_path)
    return {int(value) for value in payload.get("outer_block_subjects", [])}


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "subject"


def _audit_fold(arrays: Any, target_subject: int) -> dict[str, Any]:
    indices = build_loso_fold_indices(arrays, target_subject)
    train_subjects = set(int(value) for value in arrays.subjects[indices.train])
    validation_subjects = set(int(value) for value in arrays.subjects[indices.validation])
    test_subjects = set(int(value) for value in arrays.subjects[indices.test])
    known = set(int(value) for value in arrays.unique_subjects)
    checks = {
        "train_excludes_target": target_subject not in train_subjects,
        "validation_excludes_target": target_subject not in validation_subjects,
        "test_contains_only_target": test_subjects == {target_subject},
        "sources_are_all_non_targets": set(indices.source_subjects) == known - {target_subject},
    }
    if not all(checks.values()):
        raise RuntimeError(f"LOSO isolation audit failed: {checks}")
    return {
        **checks,
        "source_subjects": len(indices.source_subjects),
        "train_samples": len(indices.train),
        "validation_samples": len(indices.validation),
        "test_samples": len(indices.test),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.learning_rate is not None and args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than zero")

    spec = architecture_from_args(args)
    config = PainNASConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        loso_max_epochs=args.max_epochs,
        loso_patience=args.patience,
        bootstrap_samples=args.bootstrap_samples,
        max_parameters=args.max_parameters,
    )
    devices = require_gpu(allow_cpu=bool(args.allow_cpu or args.dry_run))
    arrays = load_biovid_binary(str(args.data_dir.expanduser().resolve()), config)
    target_subject, fold_index, target_key = resolve_target(arrays, args)

    audit = _audit_fold(arrays, target_subject)
    reset_runtime(args.seed + fold_index)
    model = build_early_fusion_model(
        spec,
        input_shape=(arrays.num_modalities, arrays.sequence_length, 1),
        num_classes=config.num_classes,
    )
    parameter_count = int(model.count_params())
    del model
    if parameter_count > config.max_parameters:
        raise ValueError(
            f"Model has {parameter_count:,} parameters, exceeding --max-parameters "
            f"{config.max_parameters:,}"
        )

    target_dir = (
        args.output_dir.expanduser().resolve()
        / f"fold_{fold_index:03d}_{_safe_key(target_key)}"
    )
    setup = {
        "architecture": spec.to_dict(),
        "parameter_count": parameter_count,
        "target_subject": target_subject,
        "target_subject_key": target_key,
        "target_fold": fold_index,
        "gpu_devices": devices,
        "output_dir": str(target_dir),
        "isolation_audit": audit,
    }
    print(json.dumps(to_jsonable(setup), indent=2, sort_keys=True))
    if args.dry_run:
        print("Dry run complete; model fitting was skipped.")
        return 0

    summary = run_loso(
        arrays,
        spec,
        config,
        target_dir,
        resume=bool(args.resume),
        start_index=fold_index,
        stop_index=fold_index,
        max_folds=1,
        verbose=args.verbose,
    )
    result_path = target_dir / "folds" / f"fold_{fold_index:03d}.json"
    result = read_json(result_path)
    report = {
        **setup,
        "epochs_ran": result["epochs_ran"],
        "best_epoch": result["best_epoch"],
        "metrics": result["metrics"],
        "result_path": str(result_path),
        "summary": summary,
    }
    print(json.dumps(to_jsonable(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
