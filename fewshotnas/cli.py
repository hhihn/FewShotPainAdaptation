from __future__ import annotations

import argparse
import json
from pathlib import Path

from fewshotnas.config import FewShotNASConfig
from fewshotnas.search import run_all, run_refit, run_search
from painnas.io import to_jsonable
from painnas.runtime import require_gpu


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fixed-split BioVid CrossMod-CAN NAS")
    parser.add_argument("command", choices=("search", "refit", "all"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/fewshotnas/run_001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--tasks-per-epoch", type=int, default=10_000)
    parser.add_argument("--support-repeats", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    require_gpu(allow_cpu=args.allow_cpu)
    config = FewShotNASConfig(
        seed=args.seed, n_trials=args.n_trials, max_epochs=args.max_epochs,
        tasks_per_epoch=args.tasks_per_epoch, support_repeats=args.support_repeats,
    )
    output = Path(args.output_dir).resolve()
    if args.command == "search":
        result = run_search(args.data_dir, config, output / "search", resume=args.resume)
    elif args.command == "refit":
        result = run_refit(
            args.data_dir, config, output / "search", output / "refit",
            resume=args.resume,
        )
    else:
        result = run_all(args.data_dir, config, output, resume=args.resume)
    print(json.dumps(to_jsonable(result), indent=2, sort_keys=True))
