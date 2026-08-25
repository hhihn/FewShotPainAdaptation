"""One-time Optuna neural architecture search for the early-fusion CNN."""

from __future__ import annotations

import gc
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np
import optuna
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow import keras

from painnas.config import PROTOCOL_WARNING, PainNASConfig
from painnas.data import (
    BioVidArrays,
    GLOBAL_SEARCH_TEST_FRACTION,
    GLOBAL_SEARCH_VALIDATION_FRACTION,
    compute_source_normalization,
    deterministic_global_search_subject_split,
    indices_for_subjects,
    make_tf_dataset,
)
from painnas.io import atomic_write_csv, atomic_write_json, ensure_manifest, read_json
from painnas.model import (
    ArchitectureSpec,
    LateBranchSpec,
    LateFusionArchitectureSpec,
    ModelSpec,
    aggregate_probabilities,
    architecture_from_dict,
    build_model,
    compile_model,
    early_stopping_callbacks,
    target_output_names,
    validation_monitor,
)
from painnas.runtime import reset_runtime


STUDY_NAME = "painnas_early_fusion_binary"
SEARCH_SPACE_VERSION = 4
GLOBAL_SEARCH_PROTOCOL_VERSION = 2
GLOBAL_SEARCH_PROTOCOL_DESCRIPTION = (
    "Global architecture search uses deterministic, pairwise subject-disjoint "
    "train, validation, and architecture-selection test sets. Validation controls "
    "early stopping and pruning; the unweighted mean test-subject macro-F1 ranks "
    "completed trials. The subsequent 87-fold LOSO result remains exploratory "
    "because the selection-test subjects influence the chosen architecture."
)


def baseline_trial_parameters(fusion_mode: str = "early") -> dict[str, Any]:
    if fusion_mode == "late":
        parameters: dict[str, Any] = {"learning_rate": 1e-5}
        for modality, kernel in (("eda", 3), ("emg_ecg", 11)):
            parameters.update({
                f"{modality}_num_blocks": 7,
                **{f"{modality}_block_{index}_repeats": 1 for index in range(1, 8)},
                f"{modality}_width_multiplier": 1.0,
                f"{modality}_temporal_kernel_size": kernel,
                f"{modality}_head_type": "flatten",
                f"{modality}_dense_depth": 2,
                f"{modality}_dense_1_units": 1024,
                f"{modality}_dense_2_units": 512,
            })
        return parameters
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
        "head_type": "flatten",
        "convolution_type": "standard",
        "normalization_type": "batch",
        "pooling_type": "max",
        "pooling_size": 2,
    }


def _late_branch_from_parameters(parameters: Mapping[str, Any], modality: str) -> LateBranchSpec:
    num_blocks = int(parameters[f"{modality}_num_blocks"])
    dense_depth = int(parameters[f"{modality}_dense_depth"])
    dense_units = [int(parameters[f"{modality}_dense_1_units"])]
    if dense_depth == 2:
        dense_units.append(int(parameters[f"{modality}_dense_2_units"]))
    return LateBranchSpec(
        num_blocks=num_blocks,
        conv_repeats=tuple(int(parameters[f"{modality}_block_{index}_repeats"]) for index in range(1, num_blocks + 1)),
        width_multiplier=float(parameters[f"{modality}_width_multiplier"]),
        temporal_kernel_size=int(parameters[f"{modality}_temporal_kernel_size"]),
        dense_units=tuple(dense_units),
        head_type=str(parameters[f"{modality}_head_type"]),
    )


def architecture_from_parameters(parameters: Mapping[str, Any], fusion_mode: str = "early") -> ModelSpec:
    if fusion_mode == "late":
        shared_emg_ecg = _late_branch_from_parameters(parameters, "emg_ecg")
        return LateFusionArchitectureSpec(
            eda=_late_branch_from_parameters(parameters, "eda"),
            emg=shared_emg_ecg,
            ecg=shared_emg_ecg,
            learning_rate=float(parameters["learning_rate"]),
        )
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
        head_type=str(parameters.get("head_type", "flatten")),
        convolution_type=str(parameters.get("convolution_type", "standard")),
        normalization_type=str(parameters.get("normalization_type", "batch")),
        pooling_type=str(parameters.get("pooling_type", "max")),
        pooling_size=int(parameters.get("pooling_size", 2)),
    )


def _suggest_late_branch(trial: optuna.Trial, modality: str, kernels: list[int]) -> None:
    num_blocks = trial.suggest_int(f"{modality}_num_blocks", 5, 7)
    for index in range(1, num_blocks + 1):
        trial.suggest_categorical(f"{modality}_block_{index}_repeats", [1, 2])
    trial.suggest_categorical(f"{modality}_width_multiplier", [0.5, 1.0, 2.0])
    trial.suggest_categorical(f"{modality}_temporal_kernel_size", kernels)
    trial.suggest_categorical(f"{modality}_head_type", ["flatten", "global_average"])
    depth = trial.suggest_int(f"{modality}_dense_depth", 1, 2)
    first = trial.suggest_categorical(f"{modality}_dense_1_units", [128, 256, 512, 1024])
    if depth == 2:
        second = trial.suggest_categorical(f"{modality}_dense_2_units", [128, 256, 512])
        if second > first:
            raise optuna.TrialPruned("Dense widths must be non-increasing")


def suggest_architecture(trial: optuna.Trial, fusion_mode: str = "early") -> ModelSpec:
    if fusion_mode == "late":
        _suggest_late_branch(trial, "eda", [3, 5, 7])
        _suggest_late_branch(trial, "emg_ecg", [7, 11, 15])
        parameters = dict(trial.params)
        parameters["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        return architecture_from_parameters(parameters, fusion_mode="late")
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
    parameters["head_type"] = trial.suggest_categorical(
        "head_type", ["flatten", "global_average"]
    )
    parameters["convolution_type"] = trial.suggest_categorical(
        "convolution_type", ["standard", "separable"]
    )
    parameters["normalization_type"] = trial.suggest_categorical(
        "normalization_type", ["batch", "group", "layer"]
    )
    parameters["pooling_type"] = trial.suggest_categorical(
        "pooling_type", ["max", "average"]
    )
    parameters["pooling_size"] = trial.suggest_categorical("pooling_size", [2, 4])
    parameters["dense_depth"] = trial.suggest_int("dense_depth", 1, 2)
    parameters["dense_1_units"] = trial.suggest_categorical(
        "dense_1_units", [128, 256, 512, 1024]
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


def _trial_seed(effective_seed: int, trial_number: int, *, fixed: bool) -> int:
    return int(effective_seed) if fixed else int(effective_seed) + int(trial_number)


def _subject_macro_f1_metrics(
    arrays: BioVidArrays,
    indices: np.ndarray,
    probabilities: np.ndarray,
    subjects: Iterable[int],
    *,
    num_classes: int,
) -> dict[str, Any]:
    """Return equally weighted subject-level macro-F1 selection metrics."""

    selected_subjects = tuple(int(subject) for subject in subjects)
    predictions = np.argmax(probabilities, axis=1).astype(np.int32)
    y_true = arrays.y[indices]
    sample_subjects = arrays.subjects[indices]
    labels = np.arange(num_classes, dtype=np.int32)
    rows: list[dict[str, Any]] = []
    for subject in selected_subjects:
        mask = sample_subjects == subject
        if not np.any(mask):
            raise RuntimeError(
                f"Architecture-selection test subject {subject} has no samples"
            )
        rows.append(
            {
                "subject": subject,
                "subject_key": arrays.subject_keys.get(subject, str(subject)),
                "sample_count": int(np.sum(mask)),
                "macro_f1": float(
                    f1_score(
                        y_true[mask],
                        predictions[mask],
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
    values = np.asarray([row["macro_f1"] for row in rows], dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "pooled": float(
            f1_score(
                y_true,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "subjects": rows,
    }


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
    spec = architecture_from_parameters(
        best.params, str(search_context.get("fusion_mode", "early"))
    )
    payload = {
        "architecture": spec.to_dict(),
        "best_trial_number": int(best.number),
        "best_validation_macro_f1": float(
            best.user_attrs.get("best_validation_macro_f1", best.value)
        ),
        "parameter_count": int(best.user_attrs.get("parameter_count", -1)),
        "best_epoch": int(best.user_attrs.get("best_epoch", -1)),
        "protocol_note": protocol_note,
        **dict(search_context),
    }
    if "test_subject_macro_f1_mean" in best.user_attrs:
        payload.update(
            {
                "best_test_subject_macro_f1_mean": float(
                    best.user_attrs["test_subject_macro_f1_mean"]
                ),
                "best_test_subject_macro_f1_std": float(
                    best.user_attrs["test_subject_macro_f1_std"]
                ),
                "best_test_pooled_macro_f1": float(
                    best.user_attrs["test_pooled_macro_f1"]
                ),
                "best_test_subject_scores": best.user_attrs[
                    "test_subject_macro_f1_scores"
                ],
            }
        )
    atomic_write_json(output_dir / "best_architecture.json", payload)


def load_architecture(path: Path) -> ModelSpec:
    payload = read_json(path)
    architecture_payload = payload.get("architecture", payload)
    return architecture_from_dict(architecture_payload)


def load_selected_training_epochs(path: Path) -> int:
    """Load the winning trial's subject-disjoint validation-best epoch."""

    payload = read_json(path)
    value = payload.get("best_epoch")
    if isinstance(value, bool):
        raise ValueError(f"Invalid best_epoch in selected architecture: {path}")
    try:
        training_epochs = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Selected architecture does not contain a valid best_epoch: {path}"
        ) from error
    if training_epochs <= 0 or float(value) != float(training_epochs):
        raise ValueError(f"Invalid best_epoch in selected architecture: {path}")
    return training_epochs


def run_search(
    arrays: BioVidArrays,
    config: PainNASConfig,
    output_dir: Path,
    *,
    resume: bool,
    verbose: int = 1,
    train_subjects: Iterable[int] | None = None,
    validation_subjects: Iterable[int] | None = None,
    test_subjects: Iterable[int] | None = None,
    outer_target_subject: int | None = None,
    search_seed: int | None = None,
    study_name: str | None = None,
    protocol_note: str | None = None,
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
    if test_subjects is not None and train_subjects is None:
        raise ValueError(
            "test_subjects may only be provided with train_subjects and "
            "validation_subjects"
        )
    if train_subjects is None:
        global_split = deterministic_global_search_subject_split(
            arrays.unique_subjects, seed=effective_seed
        )
        train_subjects = global_split.train_subjects
        validation_subjects = global_split.validation_subjects
        test_subjects = global_split.test_subjects
    else:
        train_subjects = tuple(sorted(int(value) for value in train_subjects))
        validation_subjects = tuple(
            sorted(int(value) for value in validation_subjects or ())
        )
        test_subjects = (
            None
            if test_subjects is None
            else tuple(sorted(int(value) for value in test_subjects))
        )
    for name, values in (
        ("train", train_subjects),
        ("validation", validation_subjects),
        ("test", test_subjects),
    ):
        if values is not None and len(values) != len(set(values)):
            raise ValueError(f"Search {name} subjects must not contain duplicates")
    if not train_subjects or not validation_subjects:
        raise ValueError("Search train and validation subject sets must be non-empty")
    selection_test_enabled = test_subjects is not None
    if selection_test_enabled and not test_subjects:
        raise ValueError("Search test subject set must be non-empty")
    subject_groups = {
        "train": set(train_subjects),
        "validation": set(validation_subjects),
    }
    if selection_test_enabled:
        subject_groups["test"] = set(test_subjects or ())
    group_names = tuple(subject_groups)
    for index, left_name in enumerate(group_names):
        for right_name in group_names[index + 1 :]:
            if subject_groups[left_name] & subject_groups[right_name]:
                raise ValueError(
                    f"Search {left_name} and {right_name} subjects must be disjoint"
                )
    known_subjects = set(int(value) for value in arrays.unique_subjects)
    selected_subjects = set().union(*subject_groups.values())
    if not selected_subjects <= known_subjects:
        raise ValueError("Search subject sets contain an unknown subject")
    if selection_test_enabled and selected_subjects != known_subjects:
        raise ValueError(
            "Global three-way search subject sets must exactly cover all subjects"
        )
    if selection_test_enabled:
        expected_split = deterministic_global_search_subject_split(
            arrays.unique_subjects, seed=effective_seed
        )
        expected_groups = {
            "train": set(expected_split.train_subjects),
            "validation": set(expected_split.validation_subjects),
            "test": set(expected_split.test_subjects),
        }
        if subject_groups != expected_groups:
            raise ValueError(
                "Global three-way search subjects must match the deterministic "
                "two-stage 80/20 split for the search seed"
            )
    if outer_target_subject is not None and int(outer_target_subject) in selected_subjects:
        raise ValueError("Outer target subject leaked into architecture search")
    fixed_trial_seed = selection_test_enabled
    if protocol_note is None:
        protocol_note = (
            GLOBAL_SEARCH_PROTOCOL_DESCRIPTION
            if selection_test_enabled
            else PROTOCOL_WARNING
        )
    search_context = {
        "fusion_mode": config.fusion_mode,
        "outer_target_subject": (
            None if outer_target_subject is None else int(outer_target_subject)
        ),
        "train_subjects": train_subjects,
        "validation_subjects": validation_subjects,
        "search_seed": effective_seed,
    }
    if selection_test_enabled:
        search_context.update(
            {
                "test_subjects": test_subjects,
                "test_usage": "architecture_selection",
                "selection_objective": "mean_test_subject_macro_f1",
                "trial_seed_policy": "fixed",
                "trial_seed": effective_seed,
                "test_fraction": GLOBAL_SEARCH_TEST_FRACTION,
                "validation_fraction_of_development": (
                    GLOBAL_SEARCH_VALIDATION_FRACTION
                ),
            }
        )
    study_name = study_name or (
        f"painnas_{config.fusion_mode}_global_55_14_18"
        if selection_test_enabled
        else (
            STUDY_NAME
            if config.fusion_mode == "early"
            else "painnas_late_fusion_binary"
        )
    )
    manifest = {
        "stage": "search",
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "search_space_version": SEARCH_SPACE_VERSION,
        "study_name": study_name,
        "protocol_note": protocol_note,
        **search_context,
    }
    if selection_test_enabled:
        manifest["search_protocol_version"] = GLOBAL_SEARCH_PROTOCOL_VERSION
    ensure_manifest(output_dir / "manifest.json", manifest, resume=resume)

    train_indices = indices_for_subjects(
        arrays, train_subjects, split_code=arrays.train_split_code
    )
    validation_indices = indices_for_subjects(
        arrays, validation_subjects, split_code=arrays.test_split_code
    )
    test_indices = (
        indices_for_subjects(
            arrays, test_subjects or (), split_code=arrays.test_split_code
        )
        if selection_test_enabled
        else None
    )
    split_indices = {"train": train_indices, "validation": validation_indices}
    if test_indices is not None:
        split_indices["test"] = test_indices
    empty_splits = [name for name, values in split_indices.items() if not len(values)]
    if empty_splits:
        raise ValueError(
            "Search contains empty sample splits: " + ", ".join(empty_splits)
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
        study.enqueue_trial(baseline_trial_parameters(config.fusion_mode))

    input_shape = (arrays.num_modalities, arrays.sequence_length, 1)

    def objective(trial: optuna.Trial) -> float:
        trial_start = time.perf_counter()
        current_trial_seed = _trial_seed(
            effective_seed, int(trial.number), fixed=fixed_trial_seed
        )
        reset_runtime(current_trial_seed)
        spec = suggest_architecture(trial, config.fusion_mode)
        model = build_model(spec, input_shape=input_shape, num_classes=config.num_classes, modalities=config.modalities)
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
            seed=current_trial_seed,
            target_names=target_output_names(spec),
        )
        validation_dataset = make_tf_dataset(
            arrays,
            validation_indices,
            mean=mean,
            std=std,
            batch_size=config.batch_size,
            training=False,
            seed=effective_seed,
            target_names=target_output_names(spec),
        )
        callbacks = early_stopping_callbacks(
            monitor=validation_monitor(spec), patience=config.search_patience
        )
        callbacks.append(OptunaPruningCallback(trial, monitor=validation_monitor(spec)))
        test_dataset = None
        try:
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=config.search_max_epochs,
                callbacks=callbacks,
                verbose=verbose,
            ).history
            scores = np.asarray(history.get(validation_monitor(spec), []), dtype=np.float64)
            if scores.size == 0 or not np.any(np.isfinite(scores)):
                raise optuna.TrialPruned("No finite validation macro-F1 was produced")
            best_index = int(np.nanargmax(scores))
            best_value = float(scores[best_index])
            trial.set_user_attr("best_epoch", best_index + 1)
            trial.set_user_attr("best_validation_macro_f1", best_value)
            trial.set_user_attr("trial_seed", current_trial_seed)
            objective_value = best_value
            if test_indices is not None:
                test_dataset = make_tf_dataset(
                    arrays,
                    test_indices,
                    mean=mean,
                    std=std,
                    batch_size=config.batch_size,
                    training=False,
                    seed=effective_seed,
                    target_names=target_output_names(spec),
                )
                probabilities = np.asarray(
                    aggregate_probabilities(
                        model, model.predict(test_dataset, verbose=0)
                    )
                )
                test_metrics = _subject_macro_f1_metrics(
                    arrays,
                    test_indices,
                    probabilities,
                    test_subjects or (),
                    num_classes=config.num_classes,
                )
                objective_value = float(test_metrics["mean"])
                trial.set_user_attr(
                    "test_subject_macro_f1_mean", test_metrics["mean"]
                )
                trial.set_user_attr(
                    "test_subject_macro_f1_std", test_metrics["std"]
                )
                trial.set_user_attr(
                    "test_pooled_macro_f1", test_metrics["pooled"]
                )
                trial.set_user_attr(
                    "test_subject_macro_f1_scores", test_metrics["subjects"]
                )
            trial.set_user_attr("elapsed_seconds", time.perf_counter() - trial_start)
            return objective_value
        finally:
            del callbacks, train_dataset, validation_dataset, test_dataset, model
            reset_runtime(current_trial_seed + 1_000_000)

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
