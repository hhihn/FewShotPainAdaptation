"""Run one named PainMonit experiment in a fresh process.

Why this exists
---------------
Driving runs from notebook cells has repeatedly gone wrong in Colab, because two
things go stale independently and neither is visible from the output:

  * ``git reset --hard`` updates files on disk but cannot touch modules already
    in ``sys.modules``, so the kernel keeps running pre-pull code;
  * it also cannot touch the notebook open in the browser, so old cells run
    against new code.

A subprocess has neither problem. Every import is resolved from disk at start,
so ``git pull`` followed by this script always runs exactly the checked-out
commit. The experiment definitions live here, in version control, rather than in
hand-edited notebook cells.

Usage
-----
    python scripts/run_experiment.py D
    python scripts/run_experiment.py D --dry-run           # print config, run nothing
    python scripts/run_experiment.py D --folds 1 52        # full LOSO
    python scripts/run_experiment.py D --set num_epochs=2

``EXPERIMENTS`` below is the record of the A-H window/sensor/encoder
comparison. main.ipynb no longer imports from it: the notebook now runs the
settled PainMonit configuration (H) directly, alongside BioVid and
SenseEmotion. Keep this script for reproducing any of the earlier runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.full_loso_trial import build_parser, run_full_loso_trial  # noqa: E402

# Shared PainMonit configuration: only what differs from the CLI defaults.
BASE = {
    "dataset_source": "painmonit",
    "data_variant": "real",
    "seed": 42,
    "k_shot": 4,
    "q_query": 4,
    "task_class_ids": "0,5",
    "task_construction_mode": "single_subject",
    "normalize_mode": "support",
    "encoder_backend": "eegnet",
    "can_attention_temperature": 0.025,
    "can_meta_hidden_dim": 32,
    "can_local_loss_weight": 1.0,
    "can_margin_loss_weight": 0.5,
    "can_margin_target": 0.5,
    "can_support_mode": "learned_prototype_memory",
    "learned_prototype_slots_per_class": 2,
    "prototype_bank_init_samples_per_class": 256,
    "prototype_finetune_epochs": 4,
    "prototype_finetune_tasks_per_epoch": 500,
    "prototype_phase2_loss_mode": "ce_can",
    "source_subject_prototype_vote_use_base_index": True,
    "source_subject_prototype_vote_query_normalize_with_subject_stats": True,
    "source_subject_prototype_vote_softmax_scope": "global",
    "learning_rate": 0.0006,
    "lr_schedule": "cosine",
    "lr_decay_alpha": 0.1,
    "eegnet_temporal_filters": 8,
    "eegnet_depth_multiplier": 2,
    "eegnet_separable_filters": 16,
    "eegnet_temporal_kernel_size": 64,
    "eegnet_separable_kernel_size": 16,
    "eegnet_pool_size_1": 4,
    "eegnet_pool_size_2": 8,
    "eegnet_dropout_rate": 0.25,
    "eegnet_l2_weight": 0.0001,
    "gaussian_noise_std": 0.01,
    "num_epochs": 1,
    "tasks_per_epoch": 20000,
    "task_batch_size": 16,
    "task_chunk_size": 16,
    "val_tasks": 50,
    "heldout_eval_tasks": 500,
    "k_shot_adaptation_steps": 0,
    "train_log_every": 125,
    "eval_log_every": 125,
    "val_batch_size": 50,
    "val_every_n_train_steps": 125,
    "validation_checkpoint_metric": "task_loss",
    "validation_checkpoint_mode": "auto",
    "logging_verbosity": 1,
}

# Deltas from BASE only. PainMonit raw channels: 0=Bvp 1=Eda_E4 2=Resp
# 3=Eda_RB 4=Ecg 5=Emg.
EXPERIMENTS = {
    # Eda_RB swap, original 4 s windowing. Result: no better than baseline.
    "A": {"sensor_idx": "3,4,5"},
    # No windowing: the model sees the full 10 s signal.
    #
    # The prototype bank needs samples_per_slot * slots = 256 * 2 = 512 samples
    # per class. With windowing on, 360 real trials become 360 * 11 windows and
    # that is never binding; with windowing off only the 360 remain (45 train
    # subjects x 8 baseline trials), so initialisation fails. 128 * 2 = 256
    # fits with margin for folds whose subjects have fewer trials.
    #
    # This is a forced deviation from D, not a free choice: without
    # augmentation the data cannot supply D's bank size. Report it alongside
    # any B vs D comparison.
    "B": {
        "sensor_idx": "3,4,5",
        "disable_window_shift": True,
        "prototype_bank_init_samples_per_class": 128,
    },
    # Both EDA sensors, on D's measured window so the only difference from D is
    # the added Eda_E4 channel. Prior is mildly negative: the all-6-channel run
    # contained both EDA channels and scored below the 3-channel one, and the
    # paper's own late fusion (91.34) is slightly below Eda_RB alone (91.93).
    "C": {
        "sensor_idx": "1,3,4,5",
        "window_seconds": 8.0,
        "window_start_min_seconds": 0.0,
        "window_start_max_seconds": 2.0,
        "window_step_seconds": 0.2,
        "window_eval_start_seconds": 2.0,
    },
    # Window geometry taken from the random-forest sweep (52-fold LOSO, Eda_RB,
    # the authors' own features, so features and classifier are held fixed):
    #   full 10 s 90.77 | 2-10 s 90.53 | 3-7 s 86.23 | 1-5 s 82.79 | 0-4 s 77.73
    #   8 s windows: 0-8 88.27 < 1-9 89.33 < 2-10 90.53
    # Length dominates, but position matters at fixed length too, and the old
    # 1-5 s default was the second-worst window tested. Eval is pinned to 2 s
    # rather than the earliest jitter start because 0-8 s costs 2.26 points.
    # Window control against the ORIGINAL baseline (CrossMod, sensors 1,4,
    # 4 s @ 1-5 s, zero-shot 0.6888): same encoder and sensors, new window only.
    # Answers whether the window alone lifts that number, independently of the
    # sensor swap and the encoder change. CrossMod forces sensor_idx=(1,4).
    "E": {
        "encoder_backend": "crossmod",
        "sensor_idx": "1,4",
        "window_seconds": 8.0,
        "window_start_min_seconds": 0.0,
        "window_start_max_seconds": 2.0,
        "window_step_seconds": 0.2,
        "window_eval_start_seconds": 2.0,
    },
    # Combines the two measured winners: CrossMod (F 0.8650 -> G 0.9072, the
    # encoder is worth +4.21 at matched sensors) and the uncropped signal
    # (D 0.8614 -> B 0.8725, windowing off is worth +1.11 at matched sensors and
    # encoder). Needs B's smaller prototype bank for the same reason B does:
    # without augmentation only 360 real trials per class exist.
    "H": {
        "encoder_backend": "crossmod",
        "sensor_idx": "3,4",
        "disable_window_shift": True,
        "prototype_bank_init_samples_per_class": 128,
    },
    # Encoder comparison at MATCHED sensors, both on D's window. CrossMod is
    # structurally two-stream (two frontends, bidirectional cross-attention), so
    # 3 channels is not available to it; (3,4) = Eda_RB + Ecg is the strongest
    # pair both encoders can take. F vs G isolates the encoder.
    "F": {
        "encoder_backend": "eegnet",
        "sensor_idx": "3,4",
        "window_seconds": 8.0,
        "window_start_min_seconds": 0.0,
        "window_start_max_seconds": 2.0,
        "window_step_seconds": 0.2,
        "window_eval_start_seconds": 2.0,
    },
    "G": {
        "encoder_backend": "crossmod",
        "sensor_idx": "3,4",
        "window_seconds": 8.0,
        "window_start_min_seconds": 0.0,
        "window_start_max_seconds": 2.0,
        "window_step_seconds": 0.2,
        "window_eval_start_seconds": 2.0,
    },
    "D": {
        "sensor_idx": "3,4,5",
        "window_seconds": 8.0,
        "window_start_min_seconds": 0.0,
        "window_start_max_seconds": 2.0,
        "window_step_seconds": 0.2,
        "window_eval_start_seconds": 2.0,
    },
}


def build_argv(settings: dict) -> list[str]:
    """Turn a settings dict into argv for the full LOSO parser.

    Each boolean is emitted in the form its action expects, and an unknown key
    raises instead of being silently dropped.
    """
    parser = build_parser()
    primary: dict[str, str] = {}
    kind: dict[str, str] = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        name = action.option_strings[0].lstrip("-").replace("-", "_")
        primary[name] = action.option_strings[0]
        kind[name] = action.__class__.__name__

    argv: list[str] = []
    for key, value in settings.items():
        if value is None:
            continue
        if key not in primary:
            raise KeyError(f"No CLI flag exists for setting {key!r}")
        if kind[key] == "_StoreTrueAction":
            if value:
                argv.append(primary[key])
        elif kind[key] == "BooleanOptionalAction":
            argv.append(
                primary[key] if value else primary[key].replace("--", "--no-", 1)
            )
        else:
            argv += [primary[key], str(value)]
    return argv


def describe(args: argparse.Namespace) -> None:
    """Print the resolved configuration, as the notebook sanity cell does."""
    from data_loaders.pain_ds_config import PainDatasetConfig

    window_kwargs = {
        config_key: getattr(args, arg_name)
        for config_key, arg_name in (
            ("window_shift_window_seconds", "window_seconds"),
            ("window_shift_start_min_seconds", "window_start_min_seconds"),
            ("window_shift_start_max_seconds", "window_start_max_seconds"),
            ("window_shift_step_seconds", "window_step_seconds"),
            ("window_shift_eval_start_seconds", "window_eval_start_seconds"),
        )
        if getattr(args, arg_name) is not None
    }
    config = PainDatasetConfig(
        dataset_source=args.dataset_source,
        encoder_backend=args.encoder_backend,
        sensor_idx=tuple(int(i) for i in str(args.sensor_idx).split(",")),
        task_class_ids=tuple(int(i) for i in str(args.task_class_ids).split(",")),
        task_normalize_mode=args.normalize_mode,
        enable_window_shift_augmentation=not args.disable_window_shift,
        **window_kwargs,
    )

    channels = ", ".join(
        f"{index}={name}"
        for index, name in zip(config.sensor_idx, config.modality_names)
    )
    print(f"  channels    : {channels}")
    print(f"  classes     : {config.task_class_ids}")
    print(f"  normalize   : {config.task_normalize_mode}")
    print(f"  folds       : {args.loso_start_index} .. {args.loso_stop_index}")

    rate = config.sampling_rate_hz
    if not config.enable_window_shift_augmentation:
        print(
            f"  windowing   : OFF -> full {config.sequence_length} samples "
            f"({config.sequence_length / rate:g}s)"
        )
        return

    width = config.window_shift_window_seconds
    low = config.window_shift_start_min_seconds
    high = config.window_shift_start_max_seconds
    offsets = len(
        range(
            int(round(low * rate)),
            int(round(high * rate)) + 1,
            int(round(config.window_shift_step_seconds * rate)),
        )
    )
    eval_start = config.window_shift_eval_start_seconds
    eval_start = low if eval_start is None else eval_start
    print(f"  windowing   : ON  -> {width:g}s window ({int(width * rate)} samples)")
    print(
        f"  train jitter: {low:g}-{high:g}s starts, "
        f"step {config.window_shift_step_seconds:g}s -> {offsets} offsets"
    )
    print(
        f"  eval window : {eval_start:g}-{eval_start + width:g}s"
        + ("" if eval_start != low else "  [earliest training start]")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "experiment", choices=sorted(EXPERIMENTS), help="Named experiment to run"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/content/drive/MyDrive/FewShotPainAdaptation/data",
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default="/content/drive/MyDrive/FewShotPainAdaptationRuns",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs=2,
        metavar=("START", "STOP"),
        default=(1, 52),
        help=(
            "1-based inclusive LOSO fold range. Defaults to the full sweep: every "
            "comparison in this project is 52-fold, and folds 1-12 are a harder "
            "than average slice (D scores 0.8349 there against 0.8614 overall), "
            "so a 12-fold number cannot be compared with the rest. Pass "
            "--folds 1 12 explicitly for a quick screen."
        ),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any setting, repeatable",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration and exit",
    )
    args = parser.parse_args()

    settings = dict(BASE)
    settings.update(EXPERIMENTS[args.experiment])
    overrides: dict = {}
    settings["data_dir"] = args.data_dir
    settings["loso_start_index"], settings["loso_stop_index"] = args.folds

    for override in args.set:
        key, separator, raw = override.partition("=")
        if not separator:
            raise SystemExit(f"--set expects KEY=VALUE, got {override!r}")
        key = key.strip()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        settings[key] = value
        overrides[key] = value

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.run_root) / f"cli-{stamp}-painmonit-exp{args.experiment}"
    run_dir.mkdir(parents=True, exist_ok=True)
    settings["output_json"] = str(run_dir / "full_loso_payload.json")
    settings["training_progress_output_dir"] = str(run_dir / "training_progress")
    settings["model_architecture_output"] = str(run_dir / "model_summary.txt")

    trial_args = build_parser().parse_args(build_argv(settings))

    print(f"=== Experiment {args.experiment} ===")
    # Show the experiment's own settings and the commit they came from. A stale
    # clone is otherwise only visible deep in the run-config dump, after the
    # run has already started.
    try:
        import subprocess

        head = subprocess.check_output(
            ["git", "-C", str(ROOT), "log", "-1", "--format=%h %s"], text=True
        ).strip()
        print(f"  commit      : {head}")
    except Exception:
        pass
    print("  settings    : " + ", ".join(
        f"{key}={value!r}" for key, value in EXPERIMENTS[args.experiment].items()
    ))
    if overrides:
        # --set values are what distinguishes a repeat from its original, so they
        # must be visible next to the experiment's own settings, not implied.
        print("  overrides   : " + ", ".join(
            f"{key}={value!r}" for key, value in overrides.items()
        ))
    print(f"  run dir     : {run_dir}")
    describe(trial_args)
    print()

    if args.dry_run:
        print("dry run: nothing executed")
        return

    payload = run_full_loso_trial(trial_args)

    # full_loso_payload.json holds everything but is far too large to read by
    # eye; write the headline numbers beside it in JSON and CSV so the result
    # survives a closed Colab session.
    summary = payload.get("summary", {})
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std"])
        writer.writerows(sorted(
            (key, stats["mean"], stats["std"])
            for key, stats in summary.items()
            if isinstance(stats, dict)
        ))
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment": args.experiment,
                "folds": summary.get("num_folds"),
                "elapsed_hours": round(payload.get("elapsed_seconds", 0) / 3600, 3),
                "config": {
                    key: payload.get("config", {}).get(key)
                    for key in (
                        "encoder_backend", "sensor_idx", "modality_names",
                        "task_class_ids", "window_shift_enabled",
                        "window_shift_window_seconds",
                        "window_shift_start_min_seconds",
                        "window_shift_start_max_seconds",
                        "window_shift_eval_start_seconds",
                        "loso_start_index", "loso_stop_index",
                    )
                },
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print("\n=== Result ===")
    for metric in (
        "zero_shot_accuracy",
        "k_shot_accuracy",
        "source_subject_prototype_vote_accuracy",
    ):
        stats = summary.get(metric)
        if stats:
            print(f"  {metric:<38} {stats['mean']:.4f} +/- {stats['std']:.4f}")
    print(f"\nArtifacts in {run_dir}")
    print("  summary.json / summary.csv hold the headline numbers")


if __name__ == "__main__":
    main()
