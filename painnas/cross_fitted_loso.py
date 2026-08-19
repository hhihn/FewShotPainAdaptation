"""Uncertainty-aware block NAS with warm-started LOSO continuation."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable

import numpy as np
import optuna
import tensorflow as tf
from tensorflow import keras

from painnas.config import CROSS_FITTED_PROTOCOL_DESCRIPTION, PainNASConfig
from painnas.data import (
    BioVidArrays,
    CrossFittedSubjectPlan,
    build_cross_fitted_subject_plan,
    compute_source_normalization,
    indices_for_subjects,
    make_tf_dataset,
)
from painnas.io import atomic_write_csv, atomic_write_json, ensure_manifest, read_json
from painnas.loso import METRIC_NAMES, classification_metrics
from painnas.model import (
    ModelSpec,
    aggregate_probabilities,
    architecture_from_dict,
    build_model,
    compile_model,
    early_stopping_callbacks,
    learned_fusion_weights,
    target_output_names,
    validation_monitor,
)
from painnas.runtime import reset_runtime
from painnas.search import (
    SEARCH_SPACE_VERSION,
    architecture_from_parameters,
    baseline_trial_parameters,
    suggest_architecture,
)


def uncertainty_aware_accuracy(
    subject_accuracies: Iterable[float], *, beta: float
) -> dict[str, float]:
    """Return subject-macro accuracy minus beta times its standard error."""

    values = np.asarray(tuple(subject_accuracies), dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("subject_accuracies must contain finite values")
    if beta < 0:
        raise ValueError("beta must be >= 0")
    mean = float(np.mean(values))
    standard_deviation = (
        float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    )
    standard_error = standard_deviation / float(np.sqrt(values.size))
    return {
        "subject_accuracy_mean": mean,
        "subject_accuracy_std": standard_deviation,
        "subject_accuracy_standard_error": standard_error,
        "uncertainty_beta": float(beta),
        "uncertainty_objective": mean - float(beta) * standard_error,
        "evaluated_subject_count": int(values.size),
    }


def _architecture_fingerprint(spec: ModelSpec) -> str:
    payload = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _round_positive_median(values: Iterable[int]) -> int:
    median = float(np.median(np.asarray(tuple(values), dtype=np.float64)))
    return max(1, int(np.floor(median + 0.5)))


def resolve_continuation_epochs(
    search_result: dict[str, Any], config: PainNASConfig
) -> int:
    """Resolve the fixed override or the search-derived continuation length."""
    if config.cross_fitted_continuation_epochs is not None:
        return int(config.cross_fitted_continuation_epochs)
    return int(search_result["median_best_epoch"])


def _format_duration(seconds: float | None) -> str:
    """Format an ETA or elapsed duration compactly for progress output."""
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    rounded = int(round(float(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds_part:02d}s"
    if minutes:
        return f"{minutes}m {seconds_part:02d}s"
    return f"{seconds_part}s"


def _continuation_eta(durations: Iterable[float], remaining_folds: int) -> float | None:
    """Estimate remaining continuation time from the robust median fold duration."""
    finite = np.asarray(
        [value for value in durations if np.isfinite(value) and value >= 0],
        dtype=np.float64,
    )
    if finite.size == 0 or remaining_folds <= 0:
        return 0.0 if remaining_folds <= 0 else None
    return float(np.median(finite)) * int(remaining_folds)


class EpochProgressCallback(keras.callbacks.Callback):
    """Print intermediate metrics and an epoch-level ETA without a noisy batch bar."""

    def __init__(
        self,
        *,
        label: str,
        total_epochs: int,
        metric_names: Iterable[str],
    ) -> None:
        super().__init__()
        self.label = label
        self.total_epochs = int(total_epochs)
        self.metric_names = tuple(metric_names)
        self.started_at = 0.0

    def on_train_begin(self, logs=None) -> None:
        del logs
        self.started_at = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        logs = logs or {}
        elapsed = time.perf_counter() - self.started_at
        completed = epoch + 1
        remaining = max(0, self.total_epochs - completed)
        eta = elapsed / completed * remaining if completed else None
        metrics = []
        for name in self.metric_names:
            value = logs.get(name)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                metrics.append(f"{name}={numeric:.4f}")
        metric_text = " | ".join(metrics) if metrics else "metrics unavailable"
        print(
            f"[{self.label}] epoch {completed}/{self.total_epochs} | {metric_text} | "
            f"elapsed {_format_duration(elapsed)} | ETA {_format_duration(eta)}",
            flush=True,
        )


def _subject_accuracy_rows(
    arrays: BioVidArrays,
    indices: np.ndarray,
    probabilities: np.ndarray,
    subjects: Iterable[int],
    *,
    inner_fold_index: int,
) -> list[dict[str, Any]]:
    predictions = np.argmax(probabilities, axis=1).astype(np.int32)
    index_subjects = arrays.subjects[indices]
    rows = []
    for subject in subjects:
        mask = index_subjects == int(subject)
        if not np.any(mask):
            raise RuntimeError(f"No validation samples for subject {subject}")
        accuracy = float(np.mean(predictions[mask] == arrays.y[indices][mask]))
        rows.append(
            {
                "inner_fold_index": int(inner_fold_index),
                "subject": int(subject),
                "subject_key": arrays.subject_keys.get(int(subject), str(subject)),
                "sample_count": int(np.sum(mask)),
                "accuracy": accuracy,
            }
        )
    return rows


class SubjectMacroAccuracyCallback(keras.callbacks.Callback):
    """Insert equal-subject validation accuracy into Keras epoch logs."""

    def __init__(
        self,
        arrays: BioVidArrays,
        validation_indices: np.ndarray,
        validation_dataset,
        validation_subjects: Iterable[int],
        *,
        inner_fold_index: int,
    ) -> None:
        super().__init__()
        self.arrays = arrays
        self.validation_indices = validation_indices
        self.validation_dataset = validation_dataset
        self.validation_subjects = tuple(int(value) for value in validation_subjects)
        self.inner_fold_index = int(inner_fold_index)
        self.values: list[float] = []

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        probabilities = np.asarray(aggregate_probabilities(
            self.model, self.model.predict(self.validation_dataset, verbose=0)
        ))
        rows = _subject_accuracy_rows(
            self.arrays,
            self.validation_indices,
            probabilities,
            self.validation_subjects,
            inner_fold_index=self.inner_fold_index,
        )
        value = float(np.mean([row["accuracy"] for row in rows]))
        self.values.append(value)
        if logs is not None:
            logs["val_subject_macro_accuracy"] = value


def _trial_rows(study: optuna.Study) -> list[dict[str, Any]]:
    rows = []
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


def _copy_checkpoint_atomically(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _persist_block_search(
    study: optuna.Study,
    output_dir: Path,
    *,
    outer_block_index: int,
    outer_block_subjects: tuple[int, ...],
    development_subjects: tuple[int, ...],
    inner_folds: tuple[tuple[int, ...], ...],
) -> None:
    atomic_write_csv(output_dir / "trials.csv", _trial_rows(study))
    complete = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not complete:
        return
    best = max(complete, key=lambda trial: float(trial.value))
    result_path = output_dir / "trials" / f"trial_{best.number:04d}" / "result.json"
    if not result_path.exists():
        raise RuntimeError(f"Missing completed-trial result: {result_path}")
    result = read_json(result_path)
    canonical_checkpoint = output_dir / "best_warmstart.weights.h5"
    checkpoint_metadata_path = output_dir / "best_warmstart_checkpoint.json"
    existing_metadata = (
        read_json(checkpoint_metadata_path) if checkpoint_metadata_path.exists() else {}
    )
    if int(existing_metadata.get("trial_number", -1)) != int(best.number):
        candidate = result_path.parent / "candidate.weights.h5"
        if not candidate.exists():
            raise RuntimeError(
                "Winning trial checkpoint is unavailable; the interrupted trial must "
                f"be rerun: {candidate}"
            )
        _copy_checkpoint_atomically(candidate, canonical_checkpoint)
        atomic_write_json(
            checkpoint_metadata_path,
            {
                "trial_number": int(best.number),
                "architecture_fingerprint": result["architecture_fingerprint"],
                **result["warm_start_checkpoint"],
            },
        )
    spec = architecture_from_dict(result["architecture"])
    atomic_write_json(
        output_dir / "best_architecture.json",
        {
            "architecture": spec.to_dict(),
            "architecture_fingerprint": result["architecture_fingerprint"],
            "best_trial_number": int(best.number),
            "best_uncertainty_objective": float(best.value),
            "best_subject_accuracy_mean": result["subject_accuracy_mean"],
            "best_subject_accuracy_std": result["subject_accuracy_std"],
            "best_subject_accuracy_standard_error": result[
                "subject_accuracy_standard_error"
            ],
            "uncertainty_beta": result["uncertainty_beta"],
            "median_best_epoch": result["median_best_epoch"],
            "parameter_count": result["parameter_count"],
            "warm_start_checkpoint": "best_warmstart.weights.h5",
            "warm_start_checkpoint_metadata": result["warm_start_checkpoint"],
            "outer_block_index": int(outer_block_index),
            "outer_block_subjects": outer_block_subjects,
            "development_subjects": development_subjects,
            "inner_folds": inner_folds,
            "protocol_description": CROSS_FITTED_PROTOCOL_DESCRIPTION,
        },
    )
    for candidate in (output_dir / "trials").glob("trial_*/candidate.weights.h5"):
        if candidate.exists():
            candidate.unlink()


def run_uncertainty_aware_block_search(
    arrays: BioVidArrays,
    config: PainNASConfig,
    output_dir: Path,
    *,
    outer_block_index: int,
    outer_block_subjects: Iterable[int],
    inner_folds: Iterable[Iterable[int]],
    resume: bool,
    verbose: int = 1,
) -> dict[str, Any]:
    """Search architectures across independent inner subject folds."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays.validate(config, require_expected_subjects=False)
    outer_block_subjects = tuple(sorted(int(value) for value in outer_block_subjects))
    inner_folds = tuple(
        tuple(sorted(int(value) for value in fold)) for fold in inner_folds
    )
    all_subjects = set(int(value) for value in arrays.unique_subjects)
    development_subjects = tuple(sorted(all_subjects - set(outer_block_subjects)))
    if len(inner_folds) != config.inner_fold_count:
        raise ValueError("Inner fold count does not match the configuration")
    flattened_inner = [subject for fold in inner_folds for subject in fold]
    if len(flattened_inner) != len(set(flattened_inner)):
        raise ValueError("Inner folds overlap")
    if set(flattened_inner) != set(development_subjects):
        raise ValueError("Inner folds must exactly cover development subjects")
    search_seed = config.seed + int(outer_block_index) * 10_000_000
    study_name = f"painnas_{config.fusion_mode}_cross_fitted_block_{outer_block_index:03d}"
    manifest = {
        "stage": "uncertainty_aware_block_search",
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "search_space_version": SEARCH_SPACE_VERSION,
        "outer_block_index": int(outer_block_index),
        "outer_block_subjects": outer_block_subjects,
        "development_subjects": development_subjects,
        "inner_folds": inner_folds,
        "study_name": study_name,
        "search_seed": search_seed,
        "protocol_description": CROSS_FITTED_PROTOCOL_DESCRIPTION,
    }
    ensure_manifest(output_dir / "manifest.json", manifest, resume=resume)
    atomic_write_json(output_dir / "inner_folds.json", manifest)
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(exist_ok=True)
    database_path = (output_dir / "study.sqlite3").resolve()
    if database_path.exists() and not resume:
        raise FileExistsError(
            f"Optuna study already exists; pass --resume to reuse it: {database_path}"
        )
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{database_path}",
        direction="maximize",
        load_if_exists=resume,
        sampler=optuna.samplers.TPESampler(
            seed=search_seed,
            n_startup_trials=min(10, max(1, config.n_trials // 5)),
        ),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1, reduction_factor=3
        ),
    )
    if not study.trials:
        study.enqueue_trial(baseline_trial_parameters(config.fusion_mode))
    input_shape = (arrays.num_modalities, arrays.sequence_length, 1)

    def objective(trial: optuna.Trial) -> float:
        trial_start = time.perf_counter()
        spec = suggest_architecture(trial, config.fusion_mode)
        if verbose:
            print(
                f"[PainNAS block {outer_block_index}/{config.outer_block_count}] "
                f"stage=architecture search | trial {trial.number + 1}/{config.n_trials} "
                f"started",
                flush=True,
            )
        architecture_fingerprint = _architecture_fingerprint(spec)
        trial_dir = trials_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(exist_ok=True)
        candidate_checkpoint = trial_dir / "candidate.weights.h5"
        completed = False
        subject_rows: list[dict[str, Any]] = []
        fold_results: list[dict[str, Any]] = []
        parameter_count: int | None = None
        best_checkpoint_accuracy = -np.inf
        best_checkpoint_metadata: dict[str, Any] | None = None
        try:
            for fold_offset, validation_subjects in enumerate(inner_folds, start=1):
                fold_seed = search_seed + int(trial.number) * 100_000 + fold_offset
                training_subjects = tuple(
                    sorted(set(development_subjects) - set(validation_subjects))
                )
                train_indices = indices_for_subjects(
                    arrays, training_subjects, split_code=arrays.train_split_code
                )
                validation_indices = indices_for_subjects(
                    arrays, validation_subjects, split_code=arrays.test_split_code
                )
                if not len(train_indices) or not len(validation_indices):
                    raise RuntimeError("Inner fold contains an empty sample split")
                if verbose:
                    print(
                        f"[PainNAS block {outer_block_index}/{config.outer_block_count}] "
                        f"stage=architecture search | trial {trial.number + 1}/{config.n_trials} | "
                        f"inner fold {fold_offset}/{len(inner_folds)} | "
                        f"fitting candidate on {len(training_subjects)} subjects | "
                        f"validating on {len(validation_subjects)} subjects",
                        flush=True,
                    )
                mean, std = compute_source_normalization(arrays.X, train_indices)
                reset_runtime(fold_seed)
                model = build_model(spec, input_shape=input_shape, num_classes=config.num_classes, modalities=config.modalities)
                current_parameter_count = int(model.count_params())
                parameter_count = current_parameter_count
                trial.set_user_attr("parameter_count", current_parameter_count)
                if current_parameter_count > config.max_parameters:
                    del model
                    raise optuna.TrialPruned(
                        f"Model has {current_parameter_count:,} parameters; limit is "
                        f"{config.max_parameters:,}"
                    )
                compile_model(model, spec)
                train_dataset = make_tf_dataset(
                    arrays, train_indices, mean=mean, std=std,
                    batch_size=config.batch_size, training=True, seed=fold_seed,
                    target_names=target_output_names(spec),
                )
                validation_dataset = make_tf_dataset(
                    arrays, validation_indices, mean=mean, std=std,
                    batch_size=config.batch_size, training=False, seed=fold_seed,
                    target_names=target_output_names(spec),
                )
                metric_callback = SubjectMacroAccuracyCallback(
                    arrays,
                    validation_indices,
                    validation_dataset,
                    validation_subjects,
                    inner_fold_index=fold_offset,
                )
                callbacks = [metric_callback]
                callbacks.extend(
                    early_stopping_callbacks(
                        monitor="val_subject_macro_accuracy",
                        patience=config.search_patience,
                    )
                )
                if verbose:
                    callbacks.append(
                        EpochProgressCallback(
                            label=(
                                f"NAS block {outer_block_index} "
                                f"trial {trial.number + 1}/{config.n_trials} "
                                f"inner fold {fold_offset}/{len(inner_folds)}"
                            ),
                            total_epochs=config.search_max_epochs,
                            metric_names=(
                                "loss",
                                "accuracy",
                                "macro_f1",
                                "val_subject_macro_accuracy",
                            ),
                        )
                    )
                try:
                    history = model.fit(
                        train_dataset,
                        epochs=config.search_max_epochs,
                        callbacks=callbacks,
                        verbose=0 if verbose else verbose,
                    ).history
                    scores = np.asarray(metric_callback.values, dtype=np.float64)
                    if scores.size == 0 or not np.any(np.isfinite(scores)):
                        raise optuna.TrialPruned(
                            "No finite subject-macro validation accuracy was produced"
                        )
                    best_epoch = int(np.nanargmax(scores)) + 1
                    probabilities = np.asarray(aggregate_probabilities(
                        model, model.predict(validation_dataset, verbose=0)
                    ))
                    rows = _subject_accuracy_rows(
                        arrays,
                        validation_indices,
                        probabilities,
                        validation_subjects,
                        inner_fold_index=fold_offset,
                    )
                    fold_accuracy = float(np.mean([row["accuracy"] for row in rows]))
                    subject_rows.extend(rows)
                    fold_result = {
                        "inner_fold_index": fold_offset,
                        "training_subjects": training_subjects,
                        "validation_subjects": validation_subjects,
                        "training_sample_count": len(train_indices),
                        "validation_sample_count": len(validation_indices),
                        "normalization": {"mean": mean, "std": std},
                        "best_epoch": best_epoch,
                        "epochs_ran": len(history.get("loss", [])),
                        "subject_macro_accuracy": fold_accuracy,
                        "history": history,
                    }
                    fold_results.append(fold_result)
                    if fold_accuracy > best_checkpoint_accuracy:
                        model.save_weights(candidate_checkpoint)
                        best_checkpoint_accuracy = fold_accuracy
                        best_checkpoint_metadata = {
                            "inner_fold_index": fold_offset,
                            "training_subjects": training_subjects,
                            "validation_subjects": validation_subjects,
                            "best_epoch": best_epoch,
                            "subject_macro_accuracy": fold_accuracy,
                            "fusion_weights": learned_fusion_weights(model),
                        }
                finally:
                    del callbacks, metric_callback, train_dataset, validation_dataset, model
                    reset_runtime(fold_seed + 50_000)
                    gc.collect()
                partial = uncertainty_aware_accuracy(
                    (row["accuracy"] for row in subject_rows),
                    beta=config.uncertainty_beta,
                )
                trial.report(partial["uncertainty_objective"], step=fold_offset)
                if trial.should_prune():
                    raise optuna.TrialPruned(
                        f"Pruned after inner fold {fold_offset}: "
                        f"objective={partial['uncertainty_objective']:.6f}"
                    )
            metrics = uncertainty_aware_accuracy(
                (row["accuracy"] for row in subject_rows),
                beta=config.uncertainty_beta,
            )
            if best_checkpoint_metadata is None or not candidate_checkpoint.exists():
                raise RuntimeError("Trial did not retain a warm-start checkpoint")
            median_best_epoch = _round_positive_median(
                result["best_epoch"] for result in fold_results
            )
            result = {
                "trial_number": int(trial.number),
                "architecture": spec.to_dict(),
                "architecture_fingerprint": architecture_fingerprint,
                "parameter_count": int(parameter_count or 0),
                **metrics,
                "median_best_epoch": median_best_epoch,
                "warm_start_checkpoint": best_checkpoint_metadata,
                "inner_fold_results": fold_results,
                "subject_accuracies": subject_rows,
                "elapsed_seconds": time.perf_counter() - trial_start,
            }
            atomic_write_json(trial_dir / "result.json", result)
            trial.set_user_attr("subject_accuracy_mean", metrics["subject_accuracy_mean"])
            trial.set_user_attr(
                "subject_accuracy_standard_error",
                metrics["subject_accuracy_standard_error"],
            )
            trial.set_user_attr("median_best_epoch", median_best_epoch)
            trial.set_user_attr("elapsed_seconds", result["elapsed_seconds"])
            completed = True
            return float(metrics["uncertainty_objective"])
        finally:
            if not completed and candidate_checkpoint.exists():
                candidate_checkpoint.unlink()
            gc.collect()

    persist = lambda current_study, _trial=None: _persist_block_search(
        current_study,
        output_dir,
        outer_block_index=outer_block_index,
        outer_block_subjects=outer_block_subjects,
        development_subjects=development_subjects,
        inner_folds=inner_folds,
    )
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
            callbacks=[persist],
            catch=(tf.errors.ResourceExhaustedError,),
            gc_after_trial=True,
        )
    persist(study)
    best_path = output_dir / "best_architecture.json"
    checkpoint_path = output_dir / "best_warmstart.weights.h5"
    if not best_path.exists() or not checkpoint_path.exists():
        raise RuntimeError(f"Block search did not produce a usable winner: {output_dir}")
    payload = read_json(best_path)
    return {
        "study_name": study_name,
        "trial_count": sum(trial.state in finished_states for trial in study.trials),
        "study_record_count": len(study.trials),
        "best_architecture_path": str(best_path),
        "warm_start_checkpoint_path": str(checkpoint_path),
        **payload,
    }


def _selected_subjects(
    subjects: np.ndarray,
    *,
    start_index: int | None,
    stop_index: int | None,
    max_folds: int | None,
) -> list[int]:
    ordered = [int(value) for value in sorted(subjects.tolist())]
    start = 1 if start_index is None else int(start_index)
    stop = len(ordered) if stop_index is None else int(stop_index)
    if start < 1 or stop < start or stop > len(ordered):
        raise ValueError(
            f"Invalid one-based fold range [{start}, {stop}] for {len(ordered)} subjects"
        )
    selected = ordered[start - 1 : stop]
    if max_folds is not None:
        if max_folds <= 0:
            raise ValueError("max_folds must be > 0")
        selected = selected[: int(max_folds)]
    return selected


def _metric_summary(
    values: Iterable[float], *, bootstrap_samples: int, seed: int
) -> dict[str, float]:
    finite = np.asarray(tuple(values), dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {key: float("nan") for key in ("mean", "std", "ci_low", "ci_high")}
    bootstrap = np.random.default_rng(seed).choice(
        finite, size=(bootstrap_samples, finite.size), replace=True
    ).mean(axis=1)
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "ci_low": float(np.percentile(bootstrap, 2.5)),
        "ci_high": float(np.percentile(bootstrap, 97.5)),
    }


def _aggregate_cross_fitted_results(
    payloads: list[dict[str, Any]],
    config: PainNASConfig,
    output_dir: Path,
    *,
    total_folds: int,
) -> dict[str, Any]:
    metric_rows, prediction_rows = [], []
    architecture_counts: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        row = {
            "fold_index": payload["fold_index"],
            "target_subject": payload["target_subject"],
            "target_subject_key": payload["target_subject_key"],
            "outer_block_index": payload["outer_block_index"],
            "selected_trial": payload["selected_trial"],
            "selected_uncertainty_objective": payload[
                "selected_uncertainty_objective"
            ],
            "selected_subject_accuracy_mean": payload[
                "selected_subject_accuracy_mean"
            ],
            "selected_subject_accuracy_standard_error": payload[
                "selected_subject_accuracy_standard_error"
            ],
            "warm_start_inner_fold_index": payload[
                "warm_start_checkpoint_metadata"
            ]["inner_fold_index"],
            "warm_start_epoch": payload["warm_start_checkpoint_metadata"][
                "best_epoch"
            ],
            "continuation_epochs": payload["continuation_epochs"],
            "continuation_epochs_ran": payload["continuation_epochs_ran"],
            "continuation_best_epoch": payload["continuation_best_epoch"],
            "continuation_validation_samples": payload[
                "continuation_validation_samples"
            ],
            "parameter_count": payload["parameter_count"],
            "architecture_fingerprint": payload["architecture_fingerprint"],
            "elapsed_seconds": payload["elapsed_seconds"],
        }
        row.update(payload["metrics"])
        row.pop("confusion_matrix", None)
        metric_rows.append(row)
        prediction_rows.extend(payload["predictions"])
        fingerprint = payload["architecture_fingerprint"]
        if fingerprint not in architecture_counts:
            architecture_counts[fingerprint] = {
                "architecture_fingerprint": fingerprint,
                "selection_count": 0,
                "loso_fold_count": 0,
                "outer_blocks": [],
                **payload["selected_architecture"],
            }
        architecture_counts[fingerprint]["loso_fold_count"] += 1
        outer_block_index = int(payload["outer_block_index"])
        if outer_block_index not in architecture_counts[fingerprint]["outer_blocks"]:
            architecture_counts[fingerprint]["outer_blocks"].append(
                outer_block_index
            )
            architecture_counts[fingerprint]["selection_count"] += 1
    atomic_write_csv(output_dir / "fold_metrics.csv", metric_rows)
    atomic_write_csv(output_dir / "predictions.csv", prediction_rows)
    atomic_write_csv(
        output_dir / "architecture_frequencies.csv",
        sorted(
            architecture_counts.values(),
            key=lambda row: (-int(row["selection_count"]), row["architecture_fingerprint"]),
        ),
    )
    summaries = {
        metric_name: _metric_summary(
            (row[metric_name] for row in metric_rows),
            bootstrap_samples=config.bootstrap_samples,
            seed=config.seed + offset,
        )
        for offset, metric_name in enumerate(METRIC_NAMES)
    }
    confusion = np.sum(
        [np.asarray(payload["metrics"]["confusion_matrix"]) for payload in payloads],
        axis=0,
    )
    summary = {
        "completed_folds": len(payloads),
        "total_folds": total_folds,
        "completed_outer_blocks": len(
            {int(payload["outer_block_index"]) for payload in payloads}
        ),
        "metrics": summaries,
        "aggregate_confusion_matrix": confusion,
        "unique_selected_architectures": len(architecture_counts),
        "protocol_description": CROSS_FITTED_PROTOCOL_DESCRIPTION,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _plan_payload(plan: CrossFittedSubjectPlan) -> dict[str, Any]:
    return {
        "seed": plan.seed,
        "outer_blocks": plan.outer_blocks,
        "inner_folds_by_block": plan.inner_folds_by_block,
    }


def run_cross_fitted_loso_nas(
    arrays: BioVidArrays,
    config: PainNASConfig,
    output_dir: Path,
    *,
    resume: bool,
    start_index: int | None = None,
    stop_index: int | None = None,
    max_folds: int | None = None,
    verbose: int = 1,
) -> dict[str, Any]:
    """Run block-level NAS and warm-started individual LOSO evaluation."""

    arrays.validate(config, require_expected_subjects=False)
    plan = build_cross_fitted_subject_plan(
        arrays.unique_subjects,
        outer_block_count=config.outer_block_count,
        inner_fold_count=config.inner_fold_count,
        seed=config.seed,
    )
    output_dir = Path(output_dir)
    blocks_dir = output_dir / "blocks"
    folds_dir = output_dir / "folds"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(parents=True, exist_ok=True)
    config_fingerprint = config.fingerprint()
    continuation_epoch_rule = (
        "fixed config.cross_fitted_continuation_epochs"
        if config.cross_fitted_continuation_epochs is not None
        else "additional rounded median inner best epoch"
    )
    manifest = {
        "stage": "cross_fitted_loso_nas",
        "config": config.to_dict(),
        "config_fingerprint": config_fingerprint,
        "subject_plan": _plan_payload(plan),
        "objective": "subject_accuracy_mean - uncertainty_beta * standard_error",
        "warm_start_rule": "highest-inner-fold-macro-accuracy checkpoint",
        "continuation_epoch_rule": continuation_epoch_rule,
        "protocol_description": CROSS_FITTED_PROTOCOL_DESCRIPTION,
    }
    ensure_manifest(output_dir / "manifest.json", manifest, resume=resume)
    atomic_write_json(output_dir / "subject_blocks.json", _plan_payload(plan))
    selected = _selected_subjects(
        arrays.unique_subjects,
        start_index=start_index,
        stop_index=stop_index,
        max_folds=max_folds,
    )
    all_subjects = [int(value) for value in sorted(arrays.unique_subjects.tolist())]
    fold_number_by_subject = {
        subject: index for index, subject in enumerate(all_subjects, start=1)
    }
    block_index_by_subject = {
        subject: block_index
        for block_index, block in enumerate(plan.outer_blocks, start=1)
        for subject in block
    }
    payload_by_fold: dict[int, dict[str, Any]] = {}
    if resume:
        for result_path in sorted(folds_dir.glob("fold_*/result.json")):
            payload = read_json(result_path)
            if payload.get("config_fingerprint") != config_fingerprint:
                raise ValueError(f"Configuration mismatch in resumed fold: {result_path}")
            payload_by_fold[int(payload["fold_index"])] = payload

    selected_fold_indices = {fold_number_by_subject[subject] for subject in selected}
    completed_selected = selected_fold_indices.intersection(payload_by_fold)
    pending_folds = len(selected_fold_indices) - len(completed_selected)
    continuation_durations = [
        float(payload["continuation_elapsed_seconds"])
        for fold_index, payload in payload_by_fold.items()
        if fold_index in selected_fold_indices
        and payload.get("continuation_elapsed_seconds") is not None
    ]
    completed_this_run = 0
    if verbose:
        initial_eta = _continuation_eta(continuation_durations, pending_folds)
        print(
            f"[PainNAS] selected LOSO folds: {len(selected_fold_indices)} | "
            f"already complete: {len(completed_selected)} | pending: {pending_folds} | "
            f"continuation ETA: {_format_duration(initial_eta)}",
            flush=True,
        )

    required_blocks = sorted({block_index_by_subject[subject] for subject in selected})
    for block_index in required_blocks:
        block = plan.outer_blocks[block_index - 1]
        inner_folds = plan.inner_folds_by_block[block_index - 1]
        block_dir = blocks_dir / f"block_{block_index:03d}"
        block_targets = [subject for subject in selected if subject in set(block)]
        completed_block_targets = sum(
            fold_number_by_subject[subject] in payload_by_fold for subject in block_targets
        )
        if verbose:
            search_action = (
                "resuming/checking saved search"
                if (block_dir / "search" / "study.sqlite3").exists()
                else "starting search"
            )
            print(
                f"\n[PainNAS block {block_index}/{config.outer_block_count}] "
                f"stage=architecture search | {search_action} | "
                f"trials={config.n_trials} | inner folds={len(inner_folds)} | "
                f"LOSO targets={len(block_targets)} "
                f"({completed_block_targets} already complete)",
                flush=True,
            )
        search_result = run_uncertainty_aware_block_search(
            arrays,
            config,
            block_dir / "search",
            outer_block_index=block_index,
            outer_block_subjects=block,
            inner_folds=inner_folds,
            resume=resume,
            verbose=verbose,
        )
        block_continuation_epochs = resolve_continuation_epochs(search_result, config)
        if verbose:
            print(
                f"[PainNAS block {block_index}/{config.outer_block_count}] "
                f"stage=architecture selected | trial="
                f"{int(search_result['best_trial_number']) + 1}/{config.n_trials} | "
                f"validation accuracy={search_result['best_subject_accuracy_mean']:.4f} | "
                f"selection objective={search_result['best_uncertainty_objective']:.4f} | "
                f"parameters={int(search_result['parameter_count']):,} | "
                f"LOSO continuation epochs={block_continuation_epochs}",
                flush=True,
            )
        for block_target_position, target_subject in enumerate(block_targets, start=1):
            fold_index = fold_number_by_subject[target_subject]
            fold_dir = folds_dir / f"fold_{fold_index:03d}"
            fold_dir.mkdir(exist_ok=True)
            result_path = fold_dir / "result.json"
            if result_path.exists():
                if not resume:
                    raise FileExistsError(
                        f"Fold output exists; pass --resume to reuse it: {result_path}"
                    )
                payload_by_fold[fold_index] = read_json(result_path)
                continue
            fold_start = time.perf_counter()
            fold_seed = config.seed + fold_index * 1_000_000
            source_subjects = tuple(
                subject for subject in all_subjects if subject != target_subject
            )
            train_indices = indices_for_subjects(
                arrays, source_subjects, split_code=arrays.train_split_code
            )
            test_indices = indices_for_subjects(
                arrays, (target_subject,), split_code=arrays.test_split_code
            )
            validation_indices = indices_for_subjects(
                arrays, source_subjects, split_code=arrays.test_split_code
            )
            if len(source_subjects) != len(all_subjects) - 1:
                raise RuntimeError("Final LOSO source set must exclude only the target")
            if target_subject in source_subjects:
                raise RuntimeError("Outer target leaked into final training subjects")
            if not len(validation_indices) or np.any(
                arrays.subjects[validation_indices] == target_subject
            ):
                raise RuntimeError("Outer target leaked into continuation validation")
            spec = architecture_from_dict(search_result["architecture"])
            mean, std = compute_source_normalization(arrays.X, train_indices)
            train_dataset = make_tf_dataset(
                arrays, train_indices, mean=mean, std=std,
                batch_size=config.batch_size, training=True, seed=fold_seed,
                target_names=target_output_names(spec),
            )
            test_dataset = make_tf_dataset(
                arrays, test_indices, mean=mean, std=std,
                batch_size=config.batch_size, training=False, seed=fold_seed,
                target_names=target_output_names(spec),
            )
            validation_dataset = make_tf_dataset(
                arrays, validation_indices, mean=mean, std=std,
                batch_size=config.batch_size, training=False, seed=fold_seed,
                target_names=target_output_names(spec),
            )
            architecture_fingerprint = _architecture_fingerprint(spec)
            if architecture_fingerprint != search_result["architecture_fingerprint"]:
                raise RuntimeError("Selected architecture fingerprint mismatch")
            checkpoint_path = block_dir / "search" / search_result[
                "warm_start_checkpoint"
            ]
            continuation_epochs = resolve_continuation_epochs(search_result, config)
            if verbose:
                print(
                    f"[PainNAS block {block_index}/{config.outer_block_count}] "
                    f"stage=fitting LOSO subject {block_target_position}/{len(block_targets)} | "
                    f"global fold={fold_index}/{len(all_subjects)} | "
                    f"target={arrays.subject_keys.get(target_subject, target_subject)} | "
                    f"continuation epochs={continuation_epochs}",
                    flush=True,
                )
            reset_runtime(fold_seed)
            model = build_model(spec, input_shape=(arrays.num_modalities, arrays.sequence_length, 1), num_classes=config.num_classes, modalities=config.modalities)
            model.load_weights(checkpoint_path)
            compile_model(model, spec)
            optimizer_initial_iterations = int(model.optimizer.iterations.numpy())
            if optimizer_initial_iterations != 0:
                raise RuntimeError("Warm-start optimizer did not start at iteration zero")
            callbacks = early_stopping_callbacks(
                monitor=validation_monitor(spec), patience=config.loso_patience
            )
            if verbose:
                callbacks.append(
                    EpochProgressCallback(
                        label=f"LOSO fold {fold_index}/{len(all_subjects)}",
                        total_epochs=continuation_epochs,
                        metric_names=(
                            ("loss", "pain_class_accuracy", "pain_class_macro_f1", validation_monitor(spec))
                            if spec.fusion_mode == "late"
                            else ("loss", "accuracy", "macro_f1", validation_monitor(spec))
                        ),
                    )
                )
            continuation_start = time.perf_counter()
            try:
                history = model.fit(
                    train_dataset,
                    validation_data=validation_dataset,
                    epochs=continuation_epochs,
                    callbacks=callbacks,
                    verbose=0 if verbose else verbose,
                ).history
                losses = np.asarray(history.get("loss", []), dtype=np.float64)
                if not 1 <= losses.size <= continuation_epochs or not np.all(
                    np.isfinite(losses)
                ):
                    raise RuntimeError(
                        f"Warm continuation for fold {fold_index} did not complete "
                        "a valid finite-loss training run"
                    )
                validation_scores = np.asarray(
                    history.get(validation_monitor(spec), []), dtype=np.float64
                )
                if validation_scores.size != losses.size or not np.any(
                    np.isfinite(validation_scores)
                ):
                    raise RuntimeError("Continuation produced no finite validation score")
                continuation_best_epoch = int(np.nanargmax(validation_scores)) + 1
                probabilities = np.asarray(aggregate_probabilities(model, model.predict(test_dataset, verbose=0)))
                y_true = arrays.y[test_indices]
                metrics = classification_metrics(y_true, probabilities)
                continuation_elapsed_seconds = time.perf_counter() - continuation_start
                predictions = np.argmax(probabilities, axis=1).astype(np.int32)
                subject_key = arrays.subject_keys.get(
                    target_subject, str(target_subject)
                )
                prediction_rows = [
                    {
                        "fold_index": fold_index,
                        "target_subject": target_subject,
                        "target_subject_key": subject_key,
                        "outer_block_index": block_index,
                        "dataset_index": int(dataset_index),
                        "true_raw_class": int(config.raw_class_ids[int(true_label)]),
                        "true_binary_class": int(true_label),
                        "predicted_raw_class": int(
                            config.raw_class_ids[int(predicted_label)]
                        ),
                        "predicted_binary_class": int(predicted_label),
                        "probability_t0": float(probability[0]),
                        "probability_t4": float(probability[1]),
                    }
                    for dataset_index, true_label, predicted_label, probability in zip(
                        test_indices, y_true, predictions, probabilities
                    )
                ]
                excluded_target_train = indices_for_subjects(
                    arrays, (target_subject,), split_code=arrays.train_split_code
                )
                payload = {
                    "fold_index": fold_index,
                    "target_subject": target_subject,
                    "target_subject_key": subject_key,
                    "outer_block_index": block_index,
                    "outer_block_subjects": block,
                    "nas_development_subjects": search_result["development_subjects"],
                    "inner_folds": search_result["inner_folds"],
                    "config_fingerprint": config_fingerprint,
                    "fold_seed": fold_seed,
                    "source_subject_count": len(source_subjects),
                    "source_subjects": source_subjects,
                    "final_train_samples": len(train_indices),
                    "continuation_validation_samples": len(validation_indices),
                    "target_test_samples": len(test_indices),
                    "target_train_samples_excluded": len(excluded_target_train),
                    "selected_trial": int(search_result["best_trial_number"]),
                    "selected_uncertainty_objective": float(
                        search_result["best_uncertainty_objective"]
                    ),
                    "selected_subject_accuracy_mean": float(
                        search_result["best_subject_accuracy_mean"]
                    ),
                    "selected_subject_accuracy_std": float(
                        search_result["best_subject_accuracy_std"]
                    ),
                    "selected_subject_accuracy_standard_error": float(
                        search_result["best_subject_accuracy_standard_error"]
                    ),
                    "uncertainty_beta": float(search_result["uncertainty_beta"]),
                    "selected_architecture": spec.to_dict(),
                    "architecture_fingerprint": architecture_fingerprint,
                    "warm_start_checkpoint": str(checkpoint_path),
                    "warm_start_checkpoint_metadata": search_result[
                        "warm_start_checkpoint_metadata"
                    ],
                    "continuation_epochs": continuation_epochs,
                    "continuation_epoch_rule": continuation_epoch_rule,
                    "continuation_epochs_ran": int(losses.size),
                    "continuation_best_epoch": continuation_best_epoch,
                    "optimizer_initial_iterations": optimizer_initial_iterations,
                    "parameter_count": int(model.count_params()),
                    "fusion_weights": learned_fusion_weights(model),
                    "final_normalization": {"mean": mean, "std": std},
                    "continuation_history": history,
                    "metrics": metrics,
                    "predictions": prediction_rows,
                    "continuation_elapsed_seconds": continuation_elapsed_seconds,
                    "elapsed_seconds": time.perf_counter() - fold_start,
                    "protocol_description": CROSS_FITTED_PROTOCOL_DESCRIPTION,
                }
                atomic_write_json(result_path, payload)
                payload_by_fold[fold_index] = payload
                continuation_durations.append(continuation_elapsed_seconds)
                completed_this_run += 1
                remaining_folds = max(0, pending_folds - completed_this_run)
                if verbose:
                    eta = _continuation_eta(continuation_durations, remaining_folds)
                    print(
                        f"[LOSO fold {fold_index}/{len(all_subjects)} complete] "
                        f"accuracy={metrics['accuracy']:.4f} | "
                        f"macro_f1={metrics['macro_f1']:.4f} | "
                        f"precision_t4={metrics['precision_t4']:.4f} | "
                        f"recall_t4={metrics['recall_t4']:.4f} | "
                        f"auroc={metrics['auroc']:.4f} | "
                        f"cross_entropy={metrics['cross_entropy']:.4f} | "
                        f"elapsed={_format_duration(continuation_elapsed_seconds)} | "
                        f"continuation ETA={_format_duration(eta)}",
                        flush=True,
                    )
            finally:
                del callbacks, train_dataset, validation_dataset, test_dataset, model
                reset_runtime(fold_seed + 500_000)
                gc.collect()
            _aggregate_cross_fitted_results(
                [payload_by_fold[key] for key in sorted(payload_by_fold)],
                config,
                output_dir,
                total_folds=len(all_subjects),
            )
        if verbose:
            completed_now = sum(
                fold_number_by_subject[subject] in payload_by_fold
                for subject in block_targets
            )
            print(
                f"[PainNAS block {block_index}/{config.outer_block_count}] "
                f"stage=block complete | LOSO targets complete="
                f"{completed_now}/{len(block_targets)}",
                flush=True,
            )
    return _aggregate_cross_fitted_results(
        [payload_by_fold[key] for key in sorted(payload_by_fold)],
        config,
        output_dir,
        total_folds=len(all_subjects),
    )
