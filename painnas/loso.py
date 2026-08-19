"""Fresh supervised training and evaluation for every BioVid LOSO fold."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow import keras

from painnas.config import PROTOCOL_WARNING, PainNASConfig
from painnas.data import (
    BioVidArrays,
    build_loso_fold_indices,
    compute_source_normalization,
    make_tf_dataset,
)
from painnas.io import atomic_write_csv, atomic_write_json, ensure_manifest, read_json
from painnas.model import (
    ModelSpec,
    aggregate_probabilities,
    build_model,
    compile_model,
    early_stopping_callbacks,
    learned_fusion_weights,
    target_output_names,
    validation_monitor,
)
from painnas.runtime import reset_runtime


METRIC_NAMES = (
    "accuracy",
    "macro_f1",
    "precision_t4",
    "recall_t4",
    "auroc",
    "cross_entropy",
)


def _architecture_fingerprint(spec: ModelSpec) -> str:
    payload = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = np.argmax(probabilities, axis=1).astype(np.int32)
    labels = np.arange(probabilities.shape[1], dtype=np.int32)
    try:
        auroc = float(
            roc_auc_score(y_true, probabilities[:, 1])
            if probabilities.shape[1] == 2
            else roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
        )
    except ValueError:
        auroc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "precision_macro": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, predictions, average="macro", zero_division=0)
        ),
        # Legacy binary field names remain available for existing summaries.
        # For multiclass runs they carry the corresponding macro metric.
        "precision_t4": float(
            precision_score(y_true, predictions, pos_label=1, zero_division=0)
            if probabilities.shape[1] == 2
            else precision_score(y_true, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "recall_t4": float(
            recall_score(y_true, predictions, pos_label=1, zero_division=0)
            if probabilities.shape[1] == 2
            else recall_score(y_true, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "auroc": auroc,
        "cross_entropy": float(log_loss(y_true, probabilities, labels=labels)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels),
    }


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
    sampled = rng.choice(
        finite, size=(bootstrap_samples, finite.size), replace=True
    ).mean(axis=1)
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "ci_low": float(np.percentile(sampled, 2.5)),
        "ci_high": float(np.percentile(sampled, 97.5)),
    }


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


def _fold_payload_to_rows(
    payloads: list[dict[str, Any]], arrays: BioVidArrays
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for payload in payloads:
        row = {
            "fold_index": payload["fold_index"],
            "target_subject": payload["target_subject"],
            "target_subject_key": payload["target_subject_key"],
            "best_epoch": payload["best_epoch"],
            "epochs_ran": payload["epochs_ran"],
            "parameter_count": payload["parameter_count"],
            "elapsed_seconds": payload["elapsed_seconds"],
        }
        row.update(payload["metrics"])
        row.pop("confusion_matrix", None)
        metric_rows.append(row)
        prediction_rows.extend(payload["predictions"])
    return metric_rows, prediction_rows


def _write_aggregate(
    payloads: list[dict[str, Any]],
    arrays: BioVidArrays,
    config: PainNASConfig,
    output_dir: Path,
) -> dict[str, Any]:
    metric_rows, prediction_rows = _fold_payload_to_rows(payloads, arrays)
    atomic_write_csv(output_dir / "fold_metrics.csv", metric_rows)
    atomic_write_csv(output_dir / "predictions.csv", prediction_rows)
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
        "metrics": summaries,
        "aggregate_confusion_matrix": confusion,
        "protocol_warning": PROTOCOL_WARNING,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _load_completed_folds(
    folds_dir: Path, *, architecture_fingerprint: str
) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    for fold_path in sorted(Path(folds_dir).glob("fold_*.json")):
        payload = read_json(fold_path)
        if payload.get("architecture_fingerprint") != architecture_fingerprint:
            raise ValueError(f"Architecture mismatch in resumed fold: {fold_path}")
        completed[int(payload["fold_index"])] = payload
    return completed


def run_loso(
    arrays: BioVidArrays,
    spec: ModelSpec,
    config: PainNASConfig,
    output_dir: Path,
    *,
    resume: bool,
    start_index: int | None = None,
    stop_index: int | None = None,
    max_folds: int | None = None,
    verbose: int = 1,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)
    arrays.validate(config, require_expected_subjects=False)
    if spec.fusion_mode != config.fusion_mode:
        raise ValueError(
            f"Architecture fusion mode {spec.fusion_mode!r} does not match "
            f"configuration mode {config.fusion_mode!r}"
        )
    architecture_fingerprint = _architecture_fingerprint(spec)
    manifest = {
        "stage": "loso",
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "architecture": spec.to_dict(),
        "architecture_fingerprint": architecture_fingerprint,
        "protocol_warning": PROTOCOL_WARNING,
    }
    ensure_manifest(output_dir / "manifest.json", manifest, resume=resume)
    selected_subjects = _fold_range(
        arrays.unique_subjects,
        start_index=start_index,
        stop_index=stop_index,
        max_folds=max_folds,
    )
    all_subjects = [int(value) for value in sorted(arrays.unique_subjects.tolist())]
    fold_number_by_subject = {
        subject: index for index, subject in enumerate(all_subjects, start=1)
    }
    payload_by_fold = (
        _load_completed_folds(
            folds_dir, architecture_fingerprint=architecture_fingerprint
        )
        if resume
        else {}
    )

    for target_subject in selected_subjects:
        fold_index = fold_number_by_subject[target_subject]
        fold_path = folds_dir / f"fold_{fold_index:03d}.json"
        if fold_path.exists():
            if not resume:
                raise FileExistsError(
                    f"Fold output exists; pass --resume to reuse it: {fold_path}"
                )
            existing = read_json(fold_path)
            if existing.get("architecture_fingerprint") != architecture_fingerprint:
                raise ValueError(f"Architecture mismatch in resumed fold: {fold_path}")
            payload_by_fold[fold_index] = existing
            continue

        fold_start = time.perf_counter()
        fold_seed = config.seed + fold_index
        reset_runtime(fold_seed)
        indices = build_loso_fold_indices(arrays, target_subject)
        mean, std = compute_source_normalization(arrays.X, indices.train)
        train_dataset = make_tf_dataset(
            arrays,
            indices.train,
            mean=mean,
            std=std,
            batch_size=config.batch_size,
            training=True,
            seed=fold_seed,
            target_names=target_output_names(spec),
        )
        validation_dataset = make_tf_dataset(
            arrays,
            indices.validation,
            mean=mean,
            std=std,
            batch_size=config.batch_size,
            training=False,
            seed=fold_seed,
            target_names=target_output_names(spec),
        )
        test_dataset = make_tf_dataset(
            arrays,
            indices.test,
            mean=mean,
            std=std,
            batch_size=config.batch_size,
            training=False,
            seed=fold_seed,
            target_names=target_output_names(spec),
        )
        model = build_model(spec, input_shape=(arrays.num_modalities, arrays.sequence_length, 1), num_classes=config.num_classes, modalities=config.modalities)
        compile_model(model, spec)
        if int(model.optimizer.iterations.numpy()) != 0:
            raise RuntimeError("Fresh fold optimizer did not start at iteration zero")
        callbacks = early_stopping_callbacks(
            monitor=validation_monitor(spec), patience=config.loso_patience
        )
        try:
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=config.loso_max_epochs,
                callbacks=callbacks,
                verbose=verbose,
            ).history
            probabilities = np.asarray(aggregate_probabilities(model, model.predict(test_dataset, verbose=0)))
            y_true = arrays.y[indices.test]
            metrics = classification_metrics(y_true, probabilities)
            validation_scores = np.asarray(
                history.get(validation_monitor(spec), []), dtype=np.float64
            )
            best_epoch = (
                int(np.nanargmax(validation_scores)) + 1
                if validation_scores.size
                else len(history.get("loss", []))
            )
            predictions = np.argmax(probabilities, axis=1).astype(np.int32)
            subject_key = arrays.subject_keys.get(target_subject, str(target_subject))
            prediction_rows = [
                {
                    "fold_index": fold_index,
                    "target_subject": target_subject,
                    "target_subject_key": subject_key,
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
                    indices.test, y_true, predictions, probabilities
                )
            ]
            payload = {
                "fold_index": fold_index,
                "target_subject": target_subject,
                "target_subject_key": subject_key,
                "fold_seed": fold_seed,
                "source_subject_count": len(indices.source_subjects),
                "train_samples": len(indices.train),
                "validation_samples": len(indices.validation),
                "test_samples": len(indices.test),
                "normalization": {"mean": mean, "std": std},
                "architecture_fingerprint": architecture_fingerprint,
                "parameter_count": int(model.count_params()),
                "fusion_weights": learned_fusion_weights(model),
                "optimizer_initial_iterations": 0,
                "epochs_ran": len(history.get("loss", [])),
                "best_epoch": best_epoch,
                "history": history,
                "metrics": metrics,
                "predictions": prediction_rows,
                "elapsed_seconds": time.perf_counter() - fold_start,
            }
            atomic_write_json(fold_path, payload)
            payload_by_fold[fold_index] = payload
        finally:
            del callbacks, train_dataset, validation_dataset, test_dataset, model
            reset_runtime(fold_seed + 1_000_000)
            gc.collect()
        _write_aggregate(
            [payload_by_fold[key] for key in sorted(payload_by_fold)],
            arrays,
            config,
            output_dir,
        )

    return _write_aggregate(
        [payload_by_fold[key] for key in sorted(payload_by_fold)],
        arrays,
        config,
        output_dir,
    )
