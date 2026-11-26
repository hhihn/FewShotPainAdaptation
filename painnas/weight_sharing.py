"""Replay and screen historical PainNAS candidates within each inner fold."""

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from painnas.config import PainNASConfig
from painnas.data import compute_source_normalization, indices_for_subjects, make_tf_dataset
from painnas.io import atomic_write_json, read_json
from painnas.model import (
    LATE_LOSS_WEIGHTS, LateFusionArchitectureSpec, ModelSpec, aggregate_probabilities,
    build_model, target_output_names,
)
from painnas.runtime import reset_runtime
from painnas.search import _late_branch_from_parameters, architecture_from_parameters


def reference_config(directory: Path) -> PainNASConfig:
    manifest = read_json(Path(directory) / "manifest.json")
    if manifest.get("stage") != "cross_fitted_loso_nas" or "weight_sharing" in manifest:
        raise ValueError("Reference must be an independent cross-fitted PainNAS run")
    settings = dict(manifest["config"])
    for key in ("raw_class_ids", "modalities"):
        settings[key] = tuple(settings[key])
    return PainNASConfig(**settings)


def _spec_from_row(row, fusion_mode):
    # The current parser casts numerical fields. Older late-fusion runs have
    # independent ECG and EMG architecture parameters.
    params = {key.removeprefix("param_"): value for key, value in row.items()
              if key.startswith("param_") and value}
    if fusion_mode == "late" and "emg_num_blocks" in params:
        return LateFusionArchitectureSpec(
            **{branch: _late_branch_from_parameters(params, branch)
               for branch in ("eda", "emg", "ecg")},
            learning_rate=float(params["learning_rate"]),
        )
    return architecture_from_parameters(params, fusion_mode)


def _file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_replays(directory, output_dir, config, plan, targets, top_k):
    directory = Path(directory).resolve()
    if directory == output_dir.resolve() or top_k < 1:
        raise ValueError("Use a separate output directory and a positive shortlist size")
    if config != reference_config(directory):
        raise ValueError("Weight-sharing settings must match the reference run")
    manifest_path = directory / "manifest.json"
    reference_manifest = read_json(manifest_path)
    if reference_manifest["subject_plan"] != json.loads(json.dumps(plan)):
        raise ValueError("Reference subject plan differs from the current subject plan")
    replays = {}
    for index, (outer, inner) in enumerate(
        zip(plan["outer_blocks"], plan["inner_folds_by_block"]), start=1
    ):
        if targets.isdisjoint(outer):
            continue
        search = directory / "blocks" / f"block_{index:03d}" / "search"
        block_manifest = search / "manifest.json"
        trials_path = search / "trials.csv"
        block = read_json(block_manifest)
        development = sorted(subject for fold in inner for subject in fold)
        if (block["outer_block_index"] != index
                or block["config"] != reference_manifest["config"]
                or block["outer_block_subjects"] != list(outer)
                or block["development_subjects"] != development
                or block["inner_folds"] != [list(fold) for fold in inner]):
            raise ValueError(f"Reference block subject plan differs: {search}")
        specs = {}
        historical_seconds = 0.0
        with trials_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                historical_seconds += float(row.get("duration_seconds") or 0)
                if row["state"] not in {"COMPLETE", "PRUNED"}:
                    continue
                count = row.get("attr_parameter_count")
                if count and int(float(count)) > config.max_parameters:
                    continue
                try:
                    spec = _spec_from_row(row, config.fusion_mode)
                except (KeyError, ValueError):
                    if row["state"] == "PRUNED":
                        continue
                    raise ValueError(f"Invalid completed candidate in {trials_path}") from None
                specs[int(row["trial_number"])] = spec
        if not specs:
            raise ValueError(f"No usable candidates in {trials_path}")
        winner_path = search / "best_architecture.json"
        reference_winner = read_json(winner_path)["best_trial_number"] if winner_path.exists() else None
        replays[index] = Replay(specs, top_k, {
            "block_manifest_sha256": _file_hash(block_manifest),
            "trials_sha256": _file_hash(trials_path),
            "top_k": top_k,
            "reference_trial_seconds": historical_seconds,
            "reference_winner": reference_winner,
        })
    return replays, {
        "reference_dir": str(directory),
        "manifest_sha256": _file_hash(manifest_path),
        "top_k": top_k,
        "version": 1,
    }


class ConvolutionPool:
    def __init__(self):
        self.layers = {}

    def __call__(self, layer_type, name, channels, filters, kernel):
        key = (layer_type, name, channels, filters, kernel)
        if key not in self.layers:
            self.layers[key] = layer_type(
                filters, kernel, padding="same", activation="elu", name=name
            )
        return self.layers[key]


def _train_paths(models, specs, dataset, epochs, seed):
    optimizer = keras.optimizers.Adam()
    variables = {id(v): v for model in models.values() for v in model.trainable_variables}
    optimizer.build(list(variables.values()))
    cross_entropy = keras.losses.CategoricalCrossentropy()

    def make_step(model, spec):
        @tf.function(reduce_retracing=True)
        def step(inputs, targets):
            optimizer.learning_rate.assign(spec.learning_rate)
            with tf.GradientTape() as tape:
                outputs = model(inputs, training=True)
                if isinstance(spec, LateFusionArchitectureSpec):
                    loss = tf.add_n([
                        LATE_LOSS_WEIGHTS[name] * cross_entropy(targets[name], value)
                        for name, value in zip(model.output_names, outputs)
                    ])
                else:
                    loss = cross_entropy(targets, outputs)
                tf.debugging.assert_all_finite(loss, "Non-finite supernet loss")
            active = model.trainable_variables
            optimizer.apply_gradients(zip(tape.gradient(loss, active), active))

        return step

    steps = {number: make_step(model, specs[number]) for number, model in models.items()}
    rng = np.random.default_rng(seed)
    order = iter(())
    counts = dict.fromkeys(models, 0)
    for _ in range(epochs):
        for inputs, targets in dataset:
            number = next(order, None)
            if number is None:
                order = iter(rng.permutation(sorted(models)).tolist())
                number = next(order)
            steps[number](inputs, targets)
            counts[number] += 1
    if min(counts.values()) == 0:
        raise ValueError("The screening budget must provide at least one minibatch per candidate")
    return counts


@dataclass
class Replay:
    specs: dict[int, ModelSpec]
    top_k: int
    provenance: dict

    def screen(self, arrays, config, output_dir, inner_folds, development, seed, verbose):
        from painnas.cross_fitted_loso import _subject_accuracy_rows, architecture_selection_metrics

        rows = {number: [] for number in self.specs}
        screening_seconds = 0.0
        for fold, validation in enumerate(inner_folds, start=1):
            path = output_dir / "screening" / f"fold_{fold:03d}.json"
            if path.exists():
                result = read_json(path)
            else:
                start = time.perf_counter()
                fold_seed = seed + fold
                reset_runtime(fold_seed)
                training = sorted(set(development) - set(validation))
                train_idx = indices_for_subjects(arrays, training, split_code=arrays.train_split_code)
                val_idx = indices_for_subjects(arrays, validation, split_code=arrays.test_split_code)
                mean, std = compute_source_normalization(arrays.X, train_idx)
                pool = ConvolutionPool()
                models = {}
                for number, spec in sorted(self.specs.items()):
                    model = build_model(
                        spec, input_shape=(arrays.num_modalities, arrays.sequence_length, 1),
                        num_classes=config.num_classes, modalities=config.modalities,
                        convolution_factory=pool,
                    )
                    if model.count_params() <= config.max_parameters:
                        models[number] = model
                if not models:
                    raise ValueError("No replayed candidate fits the parameter cap")
                names = target_output_names(self.specs[next(iter(models))])
                data_args = dict(mean=mean, std=std, batch_size=config.batch_size,
                                 seed=fold_seed, target_names=names)
                train_data = make_tf_dataset(arrays, train_idx, training=True, **data_args)
                val_data = make_tf_dataset(arrays, val_idx, training=False, **data_args)
                counts = _train_paths(models, self.specs, train_data, config.search_max_epochs, fold_seed)
                scores = {}
                for number, model in models.items():
                    probabilities = np.asarray(aggregate_probabilities(model, model.predict(val_data, verbose=0)))
                    if not np.all(np.isfinite(probabilities)):
                        raise ValueError(f"Non-finite screening predictions for candidate {number}")
                    scores[number] = _subject_accuracy_rows(
                        arrays, val_idx, probabilities, validation, inner_fold_index=fold
                    )
                result = dict(training_subjects=training, validation_subjects=validation,
                              candidate_rows={str(number): value for number, value in scores.items()},
                              updates=counts, elapsed_seconds=time.perf_counter() - start)
                atomic_write_json(path, result)
                del models, model, pool, train_data, val_data
                if verbose:
                    print(f"[PainNAS] screened {len(scores)} candidates in inner fold {fold}", flush=True)
            screening_seconds += result["elapsed_seconds"]
            for number, fold_rows in result["candidate_rows"].items():
                rows[int(number)].extend(fold_rows)
        scores = {
            number: architecture_selection_metrics((row["accuracy"] for row in values), config)["selection_score"]
            for number, values in rows.items() if len(values) == len(development)
        }
        shortlist = sorted(scores, key=lambda number: (-scores[number], number))[:self.top_k]
        if not shortlist:
            raise ValueError("No replayed candidate has scores for every inner fold")
        winner = self.provenance.get("reference_winner")
        atomic_write_json(output_dir / "screening" / "shortlist.json", {
            "candidates": {str(number): spec.to_dict() for number, spec in self.specs.items()},
            "scores": scores, "shortlist": shortlist, "elapsed_seconds": screening_seconds,
            "reference_winner_in_shortlist": winner in shortlist if winner is not None else None,
        })
        return shortlist
