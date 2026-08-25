"""Command-line interface for search, LOSO, or the complete PainNAS run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from painnas.config import PainNASConfig
from painnas.cross_fitted_loso import run_cross_fitted_loso_nas
from painnas.data import load_biovid_binary
from painnas.io import to_jsonable
from painnas.loso import run_loso
from painnas.nested_loso import run_nested_loso_nas
from painnas.runtime import require_gpu
from painnas.search import load_architecture, load_selected_training_epochs, run_search


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/painnas/run_001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--search-max-epochs", type=int, default=50)
    parser.add_argument(
        "--loso-max-epochs",
        type=int,
        default=100,
        help=(
            "Maximum for nested/cross-fitted training. Global fixed-architecture "
            "LOSO loads the winning NAS best_epoch instead."
        ),
    )
    parser.add_argument(
        "--cross-fitted-continuation-epochs",
        type=int,
        default=None,
        help=(
            "Maximum post-NAS epochs for cross-fitted LOSO with early stopping. "
            "By default, use the rounded median inner best epoch as the maximum."
        ),
    )
    parser.add_argument("--search-patience", type=int, default=8)
    parser.add_argument(
        "--loso-patience",
        type=int,
        default=15,
        help=(
            "Patience for nested/cross-fitted training; unused by global fixed-"
            "epoch LOSO."
        ),
    )
    parser.add_argument(
        "--search-validation-subjects",
        type=int,
        default=17,
        help=(
            "Validation-subject count for nested search. Global search uses its "
            "fixed two-stage 80/20 subject split."
        ),
    )
    parser.add_argument("--outer-block-count", type=int, default=5)
    parser.add_argument("--inner-fold-count", type=int, default=3)
    parser.add_argument("--uncertainty-beta", type=float, default=1.0)
    parser.add_argument("--max-parameters", type=int, default=32_000_000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--fusion-mode", choices=("early", "late"), default="early",
        help="Base architecture family for NAS and LOSO (default: early).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit a CPU-only run. Intended for smoke tests, not the full defaults.",
    )
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m painnas",
        description="Supervised early-fusion NAS and BioVid LOSO evaluation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()

    subparsers.add_parser(
        "search", parents=[common], help="Run or resume the one-time Optuna search."
    )
    loso_parser = subparsers.add_parser(
        "loso", parents=[common], help="Run LOSO from a selected architecture."
    )
    loso_parser.add_argument(
        "--architecture-json",
        default=None,
        help="Defaults to <output-dir>/search/best_architecture.json.",
    )
    all_parser = subparsers.add_parser(
        "all", parents=[common], help="Run search followed by full LOSO."
    )
    nested_parser = subparsers.add_parser(
        "nested",
        parents=[common],
        help="Run one NAS and fresh refit inside every outer LOSO fold.",
    )
    nested_parser.set_defaults(n_trials=10, search_max_epochs=20)
    cross_fitted_parser = subparsers.add_parser(
        "cross-fitted",
        parents=[common],
        help="Run uncertainty-aware block NAS and warm-started LOSO.",
    )
    cross_fitted_parser.set_defaults(n_trials=10, search_max_epochs=20)
    for fold_parser in (loso_parser, all_parser, nested_parser, cross_fitted_parser):
        fold_parser.add_argument("--loso-start-index", type=int, default=None)
        fold_parser.add_argument("--loso-stop-index", type=int, default=None)
        fold_parser.add_argument(
            "--max-folds",
            type=int,
            default=None,
            help="Debug-only cap; omit for the full 87-fold run.",
        )
    return parser


def _config_from_args(args: argparse.Namespace) -> PainNASConfig:
    return PainNASConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        n_trials=args.n_trials,
        search_max_epochs=args.search_max_epochs,
        loso_max_epochs=args.loso_max_epochs,
        cross_fitted_continuation_epochs=args.cross_fitted_continuation_epochs,
        search_patience=args.search_patience,
        loso_patience=args.loso_patience,
        search_validation_subjects=args.search_validation_subjects,
        outer_block_count=args.outer_block_count,
        inner_fold_count=args.inner_fold_count,
        uncertainty_beta=args.uncertainty_beta,
        max_parameters=args.max_parameters,
        bootstrap_samples=args.bootstrap_samples,
        fusion_mode=args.fusion_mode,
    )


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    devices = require_gpu(allow_cpu=bool(args.allow_cpu))
    config = _config_from_args(args)
    output_dir = Path(args.output_dir).resolve()
    arrays = load_biovid_binary(args.data_dir, config)
    result: dict[str, Any] = {
        "command": args.command,
        "gpu_devices": devices,
        "output_dir": str(output_dir),
    }

    if args.command == "cross-fitted":
        result["cross_fitted_loso"] = run_cross_fitted_loso_nas(
            arrays,
            config,
            output_dir / "cross_fitted_loso",
            resume=bool(args.resume),
            start_index=getattr(args, "loso_start_index", None),
            stop_index=getattr(args, "loso_stop_index", None),
            max_folds=getattr(args, "max_folds", None),
            verbose=args.verbose,
        )
        return result

    if args.command == "nested":
        result["nested_loso"] = run_nested_loso_nas(
            arrays,
            config,
            output_dir / "nested_loso",
            resume=bool(args.resume),
            start_index=getattr(args, "loso_start_index", None),
            stop_index=getattr(args, "loso_stop_index", None),
            max_folds=getattr(args, "max_folds", None),
            verbose=args.verbose,
        )
        return result

    if args.command in {"search", "all"}:
        result["search"] = run_search(
            arrays,
            config,
            output_dir / "search",
            resume=bool(args.resume),
            verbose=args.verbose,
        )
    if args.command in {"loso", "all"}:
        architecture_path = (
            Path(args.architecture_json).resolve()
            if getattr(args, "architecture_json", None)
            else output_dir / "search" / "best_architecture.json"
        )
        spec = load_architecture(architecture_path)
        training_epochs = load_selected_training_epochs(architecture_path)
        result["loso"] = run_loso(
            arrays,
            spec,
            config,
            output_dir / "loso",
            resume=bool(args.resume),
            training_epochs=training_epochs,
            start_index=getattr(args, "loso_start_index", None),
            stop_index=getattr(args, "loso_stop_index", None),
            max_folds=getattr(args, "max_folds", None),
            verbose=args.verbose,
        )
    return result


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = run_command(args)
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
