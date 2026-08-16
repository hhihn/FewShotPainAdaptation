"""One-time Optuna neural architecture search for the early-fusion CNN."""

from __future__ import annotations

import gc
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np
import optuna
import tensorflow as tf
from tensorflow import keras

from painnas.config import PROTOCOL_WARNING, PainNASConfig
from painnas.data import (
    BioVidArrays,
    compute_source_normalization,
    deterministic_search_subject_split,
    indices_for_subjects,
    make_tf_dataset,
)
from painnas.io import atomic_write_csv, atomic_write_json, ensure_manifest, read_json
from painnas.model import (
    ArchitectureSpec,
    build_early_fusion_model,
    compile_model,
    early_stopping_callbacks,
)
from painnas.runtime import reset_runtime


STUDY_NAME = "painnas_early_fusion_binary"


def baseline_trial_parameters() -> dict[str, Any]:
    return {
        "num_blocks": 5,
        "block_1_repeats": 2,
        "block_2_repeats": 1,
        "block_3_repeats": 1,
        "block_4_repeats": 1,
        "block_5_repeats": 1,
        "width_multiplier": 1.0,
        "temporal_kernel_size": 11,
        "dense_depth": 2,
        "dense_1_units": 1024,
        "dense_2_units": 512,
        "learning_rate": 1e-5,
    }


def architecture_from_parameters(parameters: Mapping[str, Any]) -> ArchitectureSpec:
    num_blocks = int(parameters["num_blocks"])
    repeats = tuple(
        int(parameters[f"block_{block_index}_repeats"])
        for block_index in range(1, num_blocks + 1)
    )
    dense_depth = int(parameters["dense_depth"])
    dense_units = [int(parameters["dense_1_units"])]
    if dense_depth == 2:
        dense_units.append(int(parameters["dense_2_units"]))
    return ArchitectureSpec(
        num_blocks=num_blocks,
        conv_repeats=repeats,
        width_multiplier=float(parameters["width_multiplier"]),
        temporal_kernel_size=int(parameters["temporal_kernel_size"]),
        dense_units=tuple(dense_units),
        learning_rate=float(parameters["learning_rate"]),
    )


def suggest_architecture(trial: optuna.Trial) -> ArchitectureSpec:
    num_blocks = trial.suggest_int("num_blocks", 3, 5)
    parameters: dict[str, Any] = {"num_blocks": num_blocks}
    for block_index in range(1, num_blocks + 1):
        parameters[f"block_{block_index}_repeats"] = trial.suggest_categorical(
            f"block_{block_index}_repeats", [1, 2]
        )
    parameters["width_multiplier"] = trial.suggest_categorical(
        "width_multiplier", [0.5, 1.0, 2.0]
    )
    parameters["temporal_kernel_size"] = trial.suggest_categorical(
        "temporal_kernel_size", [7, 11, 15]
    )
    parameters["dense_depth"] = trial.suggest_int("dense_depth", 1, 2)
    parameters["dense_1_units"] = trial.suggest_categorical(
        "dense_1_units", [256, 512, 1024]
    )
    if parameters["dense_depth"] == 2:
        parameters["dense_2_units"] = trial.suggest_categorical(
            "dense_2_units", [128, 256, 512]
        )
        if parameters["dense_2_units"] > parameters["dense_1_units"]:
            raise optuna.TrialPruned("Dense widths must be non-increasing")
    parameters["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-5, 1e-3, log=True
    )
    return architecture_from_parameters(parameters)


class OptunaPruningCallback(keras.callbacks.Callback):
    """Report validation macro-F1 to Optuna without optuna-integration."""

    def __init__(self, trial: optuna.Trial, monitor: str = "val_macro_f1") -> None:
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is None:
            return
        self.trial.report(float(value), step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {epoch + 1}: {self.monitor}={value}"
            )


def _trial_rows(study: optuna.Study) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "trial_number": int(trial.number),
            "state": trial.state.name,
            "value": trial.value,
            "duration_seconds": (
                trial.duration.total_seconds() if trial.duration is not None else None
            ),
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        row.update({f"attr_{key}": value for key, value in trial.user_attrs.items()})
        rows.append(row)
    return rows


def _persist_search(
    study: optuna.Study,
    output_dir: Path,
    *,
    protocol_note: str,
    search_context: Mapping[str, Any],
) -> None:
    atomic_write_csv(output_dir / "trials.csv", _trial_rows(study))
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not complete:
        return
    best = max(complete, key=lambda trial: float(trial.value))
    spec = architecture_from_parameters(best.params)
    atomic_write_json(
        output_dir / "best_architecture.json",
        {
            "architecture": spec.to_dict(),
            "best_trial_number": int(best.number),
            "best_validation_macro_f1": float(best.value),
            "parameter_count": int(best.user_attrs.get("parameter_count", -1)),
            "best_epoch": int(best.user_attrs.get("best_epoch", -1)),
            "protocol_note": protocol_note,
            **dict(search_context),
        },
    )


def load_architecture(path: Path) -> ArchitectureSpec:
    payload = read_json(path)
    architecture_payload = payload.get("architecture", payload)
    return ArchitectureSpec.from_dict(architecture_payload)


def run_search(
    arrays: BioVidArrays,
    config: PainNASConfig,
    output_dir: Path,
    *,
    resume: bool,
    verbose: int = 1,
    train_subjects: Iterable[int] | None = None,
    validation_subjects: Iterable[int] | None = None,
    outer_target_subject: int | None = None,
    search_seed: int | None = None,
    study_name: str = STUDY_NAME,
    protocol_note: str = PROTOCOL_WARNING,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays.validate(config, require_expected_subjects=False)
    effective_seed = config.seed if search_seed is None else int(search_seed)
    if (train_subjects is None) != (validation_subjects is None):
        raise ValueError(
            "train_subjects and validation_subjects must either both be provided "
            "or both be omitted"
        )
    if train_subjects is None:
        train_subjects, validation_subjects = deterministic_search_subject_split(
            arrays.unique_subjects,
            validation_subjects=config.search_validation_subjects,
            seed=effective_seed,
        )
    else:
        train_subjects = tuple(sorted(int(value) for value in train_subjects))
        validation_subjects = tuple(
            sorted(int(value) for value in validation_subjects or ())
        )
    if not train_subjects or not validation_subjects:
        raise ValueError("Search train and validation subject sets must be non-empty")
    if set(train_subjects) & set(validation_subjects):
        raise ValueError("Search train and validation subjects must be disjoint")
    known_subjects = set(int(value) for value in arrays.unique_subjects)
    selected_subjects = set(train_subjects) | set(validation_subjects)
    if not selected_subjects <= known_subjects:
        raise ValueError("Search subject sets contain an unknown subject")
    if outer_target_subject is not None and int(outer_target_subject) in selected_subjects:
        raise ValueError("Outer target subject leaked into architecture search")
    search_context = {
        "outer_target_subject": (
            None if outer_target_subject is None else int(outer_target_subject)
        ),
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "search_seed": effective_seed,
    }
    manifest = {
        "stage": "search",
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "study_name": study_name,
        "protocol_note": protocol_note,
        **search_context,
    }
    ensure_manifest(output_dir / "manifest.json", manifest, resume=resume)

    train_indices = indices_for_subjects(
        arrays, train_subjects, split_code=arrays.train_split_code
    )
    validation_indices = indices_for_subjects(
        arrays, validation_subjects, split_code=arrays.test_split_code
    )
    mean, std = compute_source_normalization(arrays.X, train_indices)
    atomic_write_json(
        output_dir / "normalization.json", {"mean": mean, "std": std}
    )

    database_path = (output_dir / "study.sqlite3").resolve()
    if database_path.exists() and not resume:
        raise FileExistsError(
            f"Optuna study already exists; pass --resume to reuse it: {database_path}"
        )
    sampler = optuna.samplers.TPESampler(
        seed=effective_seed,
        n_startup_trials=min(10, max(1, config.n_trials // 5)),
    )
    min_resource = min(5, config.search_max_epochs)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=max(1, min_resource),
        max_resource=config.search_max_epochs,
        reduction_factor=3,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{database_path}",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=resume,
    )
    if not study.trials:
        study.enqueue_trial(baseline_trial_parameters())

    input_shape = (arrays.num_modalities, arrays.sequence_length, 1)

    def objective(trial: optuna.Trial) -> float:
        trial_start = time.perf_counter()
        reset_runtime(effective_seed + int(trial.number))
        spec = suggest_architecture(trial)
        model = build_early_fusion_model(
            spec, input_shape=input_shape, num_classes=config.num_classes
        )
        parameter_count = int(model.count_params())
        trial.set_user_attr("parameter_count", parameter_count)
        if parameter_count > config.max_parameters:
            del model
            gc.collect()
            raise optuna.TrialPruned(
                f"Model has {parameter_count:,} parameters; limit is "
                f"{config.max_parameters:,}"
            )
        compile_model(model, spec)
        train_dataset = make_tf_dataset(
            arrays,
            train_indices,
            mean=mean,
            std=std,
            batch_size=config.batch_size,
            training=True,
            seed=effective_seed + int(trial.number),
        )
        validation_dataset = make_tf_dataset(
            arrays,
            validation_indices,
            mean=mean,
            std=std,
            batch_size=config.batch_size,
            training=False,
            seed=effective_seed,
        )
        callbacks = early_stopping_callbacks(
            monitor="val_macro_f1", patience=config.search_patience
        )
        callbacks.append(OptunaPruningCallback(trial))
        try:
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=config.search_max_epochs,
                callbacks=callbacks,
                verbose=verbose,
            ).history
            scores = np.asarray(history.get("val_macro_f1", []), dtype=np.float64)
            if scores.size == 0 or not np.any(np.isfinite(scores)):
                raise optuna.TrialPruned("No finite validation macro-F1 was produced")
            best_index = int(np.nanargmax(scores))
            best_value = float(scores[best_index])
            trial.set_user_attr("best_epoch", best_index + 1)
            trial.set_user_attr("elapsed_seconds", time.perf_counter() - trial_start)
            return best_value
        finally:
            del callbacks, train_dataset, validation_dataset, model
            reset_runtime(effective_seed + int(trial.number) + 1_000_000)

    finished_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    finished_trial_count = sum(
        trial.state in finished_states for trial in study.trials
    )
    remaining_trials = max(0, config.n_trials - finished_trial_count)
    if remaining_trials:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            callbacks=[
                lambda current_study, _: _persist_search(
                    current_study,
                    output_dir,
                    protocol_note=protocol_note,
                    search_context=search_context,
                )
            ],
            catch=(tf.errors.ResourceExhaustedError,),
            gc_after_trial=True,
        )
    _persist_search(
        study,
        output_dir,
        protocol_note=protocol_note,
        search_context=search_context,
    )
    best_path = output_dir / "best_architecture.json"
    if not best_path.exists():
        raise RuntimeError(
            "Architecture search produced no completed trial. Inspect trials.csv "
            f"and start a new run after correcting the failure: {output_dir}"
        )
    best_payload = read_json(best_path)
    return {
        "study_name": study_name,
        "trial_count": len(study.trials),
        "best_architecture_path": str(output_dir / "best_architecture.json"),
        **best_payload,
    }
