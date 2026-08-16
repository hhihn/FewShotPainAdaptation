"""Nested subject-level NAS followed by a fresh refit in every LOSO fold."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
from tensorflow import keras

from painnas.config import NESTED_PROTOCOL_DESCRIPTION, PainNASConfig
from painnas.data import (
    BioVidArrays,
    build_nested_loso_fold_indices,
    compute_source_normalization,
    indices_for_subjects,
    make_tf_dataset,
)
from painnas.io import atomic_write_csv, atomic_write_json, ensure_manifest, read_json
from painnas.loso import METRIC_NAMES, classification_metrics
from painnas.model import (
    ArchitectureSpec,
    build_early_fusion_model,
    compile_model,
    early_stopping_callbacks,
)
from painnas.runtime import reset_runtime
from painnas.search import run_search


def _architecture_fingerprint(spec: ArchitectureSpec) -> str:
    payload = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fold_range(
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
    array = np.asarray(tuple(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    rng = np.random.default_rng(seed)
    bootstrap_means = rng.choice(
        finite, size=(bootstrap_samples, finite.size), replace=True
    ).mean(axis=1)
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "ci_low": float(np.percentile(bootstrap_means, 2.5)),
        "ci_high": float(np.percentile(bootstrap_means, 97.5)),
    }


def _load_completed_folds(
    folds_dir: Path, *, config_fingerprint: str
) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    for result_path in sorted(Path(folds_dir).glob("fold_*/result.json")):
        payload = read_json(result_path)
        if payload.get("config_fingerprint") != config_fingerprint:
            raise ValueError(f"Configuration mismatch in resumed fold: {result_path}")
        fold_index = int(payload["fold_index"])
        completed[fold_index] = payload
    return completed


def _aggregate_rows(
    payloads: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    architecture_counts: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        spec = payload["selected_architecture"]
        fingerprint = payload["architecture_fingerprint"]
        metric_row = {
            "fold_index": payload["fold_index"],
            "target_subject": payload["target_subject"],
            "target_subject_key": payload["target_subject_key"],
            "inner_train_subject_count": payload["inner_train_subject_count"],
            "inner_validation_subject_count": payload[
                "inner_validation_subject_count"
            ],
            "selected_trial": payload["selected_trial"],
            "selected_validation_macro_f1": payload[
                "selected_validation_macro_f1"
            ],
            "refit_epochs": payload["refit_epochs"],
            "refit_epochs_ran": payload["refit_epochs_ran"],
            "refit_best_epoch": payload["refit_best_epoch"],
            "parameter_count": payload["parameter_count"],
            "architecture_fingerprint": fingerprint,
            "num_blocks": spec["num_blocks"],
            "conv_repeats": spec["conv_repeats"],
            "width_multiplier": spec["width_multiplier"],
            "temporal_kernel_size": spec["temporal_kernel_size"],
            "dense_units": spec["dense_units"],
            "learning_rate": spec["learning_rate"],
            "search_elapsed_seconds": payload["search_elapsed_seconds"],
            "refit_elapsed_seconds": payload["refit_elapsed_seconds"],
            "elapsed_seconds": payload["elapsed_seconds"],
        }
        metric_row.update(payload["metrics"])
        metric_row.pop("confusion_matrix", None)
        metric_rows.append(metric_row)
        prediction_rows.extend(payload["predictions"])
        if fingerprint not in architecture_counts:
            architecture_counts[fingerprint] = {
                "architecture_fingerprint": fingerprint,
                "selection_count": 0,
                **spec,
            }
        architecture_counts[fingerprint]["selection_count"] += 1
    architecture_rows = sorted(
        architecture_counts.values(),
        key=lambda row: (-int(row["selection_count"]), row["architecture_fingerprint"]),
    )
    return metric_rows, prediction_rows, architecture_rows


def _write_aggregate(
    payloads: list[dict[str, Any]],
    config: PainNASConfig,
    output_dir: Path,
    *,
    total_folds: int,
) -> dict[str, Any]:
    metric_rows, prediction_rows, architecture_rows = _aggregate_rows(payloads)
    atomic_write_csv(output_dir / "fold_metrics.csv", metric_rows)
    atomic_write_csv(output_dir / "predictions.csv", prediction_rows)
    atomic_write_csv(
        output_dir / "architecture_frequencies.csv", architecture_rows
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
        "metrics": summaries,
        "aggregate_confusion_matrix": confusion,
        "unique_selected_architectures": len(architecture_rows),
        "protocol_description": NESTED_PROTOCOL_DESCRIPTION,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def run_nested_loso_nas(
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
    """Run one independent inner NAS and fresh refit per outer LOSO fold."""

    arrays.validate(config, require_expected_subjects=False)
    if config.search_validation_subjects >= len(arrays.unique_subjects) - 1:
        raise ValueError(
            "search_validation_subjects must leave at least one inner-training "
            "subject in every outer fold"
        )
    output_dir = Path(output_dir)
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    config_fingerprint = config.fingerprint()
    manifest = {
        "stage": "nested_loso_nas",
        "config": config.to_dict(),
        "config_fingerprint": config_fingerprint,
        "protocol_description": NESTED_PROTOCOL_DESCRIPTION,
        "refit_epoch_rule": "winning_inner_trial_best_epoch",
        "final_refit_samples": "all_source_subjects_predefined_train",
        "outer_evaluation_samples": "target_subject_predefined_test",
    }
    ensure_manifest(output_dir / "manifest.json", manifest, resume=resume)

    all_subjects = [int(value) for value in sorted(arrays.unique_subjects.tolist())]
    selected_subjects = _fold_range(
        arrays.unique_subjects,
        start_index=start_index,
        stop_index=stop_index,
        max_folds=max_folds,
    )
    fold_number_by_subject = {
        subject: index for index, subject in enumerate(all_subjects, start=1)
    }
    payload_by_fold = (
        _load_completed_folds(
            folds_dir, config_fingerprint=config_fingerprint
        )
        if resume
        else {}
    )

    for target_subject in selected_subjects:
        fold_index = fold_number_by_subject[target_subject]
        fold_dir = folds_dir / f"fold_{fold_index:03d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        result_path = fold_dir / "result.json"
        if result_path.exists():
            if not resume:
                raise FileExistsError(
                    f"Fold output exists; pass --resume to reuse it: {result_path}"
                )
            payload_by_fold[fold_index] = read_json(result_path)
            continue

        fold_start = time.perf_counter()
        split_seed = config.seed + fold_index * 100_000
        refit_seed = config.seed + fold_index * 1_000_000
        fold_indices = build_nested_loso_fold_indices(
            arrays,
            target_subject,
            validation_subjects=config.search_validation_subjects,
            seed=split_seed,
        )
        search_start = time.perf_counter()
        search_result = run_search(
            arrays,
            config,
            fold_dir / "search",
            resume=resume,
            verbose=verbose,
            train_subjects=fold_indices.inner_train_subjects,
            validation_subjects=fold_indices.inner_validation_subjects,
            outer_target_subject=target_subject,
            search_seed=split_seed,
            study_name=f"painnas_outer_fold_{fold_index:03d}",
            protocol_note=NESTED_PROTOCOL_DESCRIPTION,
        )
        search_elapsed = time.perf_counter() - search_start
        refit_epochs = int(search_result["best_epoch"])
        if not 1 <= refit_epochs <= config.search_max_epochs:
            raise ValueError(
                f"Invalid winning best epoch for fold {fold_index}: {refit_epochs}"
            )
        spec = ArchitectureSpec.from_dict(search_result["architecture"])
        architecture_fingerprint = _architecture_fingerprint(spec)

        reset_runtime(refit_seed)
        mean, std = compute_source_normalization(
            arrays.X, fold_indices.final_train
        )
        train_dataset = make_tf_dataset(
            arrays,
            fold_indices.final_train,
            mean=mean,
            std=std,
            batch_size=config.batch_size,
            training=True,
            seed=refit_seed,
        )
        test_dataset = make_tf_dataset(
            arrays,
            fold_indices.test,
            mean=mean,
            std=std,
            batch_size=config.batch_size,
            training=False,
            seed=refit_seed,
        )
        validation_dataset = make_tf_dataset(
            arrays,
            fold_indices.inner_validation,
            mean=mean,
            std=std,
            batch_size=config.batch_size,
            training=False,
            seed=refit_seed,
        )
        model = build_early_fusion_model(
            spec,
            input_shape=(arrays.num_modalities, arrays.sequence_length, 1),
            num_classes=config.num_classes,
        )
        compile_model(model, spec)
        optimizer_initial_iterations = int(model.optimizer.iterations.numpy())
        if optimizer_initial_iterations != 0:
            raise RuntimeError("Fresh refit optimizer did not start at iteration zero")
        callbacks = early_stopping_callbacks(
            monitor="val_macro_f1", patience=config.loso_patience
        )
        refit_start = time.perf_counter()
        try:
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=refit_epochs,
                callbacks=callbacks,
                verbose=verbose,
            ).history
            refit_losses = np.asarray(history.get("loss", []), dtype=np.float64)
            if not 1 <= refit_losses.size <= refit_epochs or not np.all(
                np.isfinite(refit_losses)
            ):
                raise RuntimeError(
                    f"Final refit for fold {fold_index} did not produce a valid "
                    "finite-loss training run"
                )
            validation_scores = np.asarray(
                history.get("val_macro_f1", []), dtype=np.float64
            )
            if validation_scores.size != refit_losses.size or not np.any(
                np.isfinite(validation_scores)
            ):
                raise RuntimeError("Final refit produced no finite validation score")
            refit_best_epoch = int(np.nanargmax(validation_scores)) + 1
            probabilities = np.asarray(model.predict(test_dataset, verbose=0))
            y_true = arrays.y[fold_indices.test]
            metrics = classification_metrics(y_true, probabilities)
            predictions = np.argmax(probabilities, axis=1).astype(np.int32)
            target_subject_key = arrays.subject_keys.get(
                target_subject, str(target_subject)
            )
            prediction_rows = [
                {
                    "fold_index": fold_index,
                    "target_subject": target_subject,
                    "target_subject_key": target_subject_key,
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
                    fold_indices.test, y_true, predictions, probabilities
                )
            ]
            refit_elapsed = time.perf_counter() - refit_start
            excluded_target_train = indices_for_subjects(
                arrays,
                (target_subject,),
                split_code=arrays.train_split_code,
            )
            payload = {
                "fold_index": fold_index,
                "target_subject": target_subject,
                "target_subject_key": target_subject_key,
                "config_fingerprint": config_fingerprint,
                "split_seed": split_seed,
                "refit_seed": refit_seed,
                "source_subject_count": len(fold_indices.source_subjects),
                "inner_train_subject_count": len(
                    fold_indices.inner_train_subjects
                ),
                "inner_validation_subject_count": len(
                    fold_indices.inner_validation_subjects
                ),
                "source_subjects": fold_indices.source_subjects,
                "inner_train_subjects": fold_indices.inner_train_subjects,
                "inner_validation_subjects": (
                    fold_indices.inner_validation_subjects
                ),
                "inner_train_samples": len(fold_indices.inner_train),
                "inner_validation_samples": len(fold_indices.inner_validation),
                "final_train_samples": len(fold_indices.final_train),
                "target_test_samples": len(fold_indices.test),
                "target_train_samples_excluded": len(excluded_target_train),
                "selected_trial": int(search_result["best_trial_number"]),
                "selected_validation_macro_f1": float(
                    search_result["best_validation_macro_f1"]
                ),
                "selected_architecture": spec.to_dict(),
                "architecture_fingerprint": architecture_fingerprint,
                "refit_epochs": refit_epochs,
                "refit_epochs_ran": int(refit_losses.size),
                "refit_best_epoch": refit_best_epoch,
                "optimizer_initial_iterations": optimizer_initial_iterations,
                "parameter_count": int(model.count_params()),
                "final_normalization": {"mean": mean, "std": std},
                "refit_history": history,
                "metrics": metrics,
                "predictions": prediction_rows,
                "search_elapsed_seconds": search_elapsed,
                "refit_elapsed_seconds": refit_elapsed,
                "elapsed_seconds": time.perf_counter() - fold_start,
                "protocol_description": NESTED_PROTOCOL_DESCRIPTION,
            }
            atomic_write_json(result_path, payload)
            payload_by_fold[fold_index] = payload
        finally:
            del callbacks, train_dataset, validation_dataset, test_dataset, model
            reset_runtime(refit_seed + 500_000)
            gc.collect()

        _write_aggregate(
            [payload_by_fold[key] for key in sorted(payload_by_fold)],
            config,
            output_dir,
            total_folds=len(all_subjects),
        )

    return _write_aggregate(
        [payload_by_fold[key] for key in sorted(payload_by_fold)],
        config,
        output_dir,
        total_folds=len(all_subjects),
    )
