from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import tensorflow as tf

from data_loaders.meta_ds_sampler import SixWayKShotSampler
from data_loaders.pain_ds_config import PainDatasetConfig
from data_loaders.pain_meta_dataset import PainMetaDataset
from fewshotnas.config import FewShotArchitectureSpec, FewShotNASConfig
from fewshotnas.data import SubjectSplit, deterministic_subject_split
from learner.few_shot_pain_learner import FewShotPainLearner
from painnas.io import atomic_write_csv, atomic_write_json, ensure_manifest, read_json


STUDY_NAME = "fewshotnas_biovid_crossmod_can"


def architecture_from_parameters(parameters: dict[str, Any]) -> FewShotArchitectureSpec:
    values = FewShotArchitectureSpec().to_dict()
    values.update(parameters)
    if values["prototype_aggregation"] == "mean":
        values["prototype_attention_temperature"] = 0.2
    if float(values["can_margin_loss_weight"]) == 0.0:
        values["can_margin_target"] = 0.3
    return FewShotArchitectureSpec.from_dict(values)


def suggest_architecture(trial: optuna.Trial) -> FewShotArchitectureSpec:
    margin_weight = trial.suggest_categorical(
        "can_margin_loss_weight", [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    aggregation = trial.suggest_categorical(
        "prototype_aggregation", ["mean", "attention"]
    )
    return FewShotArchitectureSpec(
        crossmod_num_heads=trial.suggest_categorical("crossmod_num_heads", [2, 4, 8]),
        crossmod_hidden_dim=trial.suggest_categorical(
            "crossmod_hidden_dim", [64, 128, 256]
        ),
        crossmod_num_layers=trial.suggest_int("crossmod_num_layers", 1, 3),
        crossmod_attention_dropout_rate=trial.suggest_categorical(
            "crossmod_attention_dropout_rate", [0.0, 0.1, 0.25, 0.35]
        ),
        crossmod_ff_activation=trial.suggest_categorical(
            "crossmod_ff_activation", ["relu", "gelu"]
        ),
        crossmod_fusion_mode=trial.suggest_categorical(
            "crossmod_fusion_mode",
            ["cross_attention_concat", "residual_concat", "gated_sum"],
        ),
        can_meta_depth=trial.suggest_int("can_meta_depth", 1, 3),
        can_meta_hidden_dim=trial.suggest_categorical(
            "can_meta_hidden_dim", [32, 64, 128]
        ),
        can_meta_activation=trial.suggest_categorical(
            "can_meta_activation", ["gelu", "relu"]
        ),
        can_temporal_pooling=trial.suggest_categorical(
            "can_temporal_pooling", ["mean", "attention", "gated"]
        ),
        can_attention_temperature=trial.suggest_categorical(
            "can_attention_temperature", [0.01, 0.025, 0.05, 0.1, 0.2]
        ),
        can_local_pool_temperature=trial.suggest_categorical(
            "can_local_pool_temperature", [0.05, 0.1, 0.2]
        ),
        prototype_feature_normalization=trial.suggest_categorical(
            "prototype_feature_normalization", ["none", "l2", "layer_l2"]
        ),
        prototype_aggregation=aggregation,
        prototype_attention_temperature=(
            trial.suggest_categorical(
                "prototype_attention_temperature", [0.1, 0.2, 0.5]
            )
            if aggregation == "attention"
            else 0.2
        ),
        learned_prototype_slots_per_class=trial.suggest_categorical(
            "learned_prototype_slots_per_class", [1, 2, 5, 10]
        ),
        prototype_bank_init_samples_per_class=trial.suggest_categorical(
            "prototype_bank_init_samples_per_class", [128, 256, 512, 1024]
        ),
        can_logit_scale_initial=trial.suggest_categorical(
            "can_logit_scale_initial", [5.0, 10.0, 20.0]
        ),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        lr_decay_alpha=trial.suggest_categorical("lr_decay_alpha", [0.05, 0.1, 0.2]),
        can_local_loss_weight=trial.suggest_categorical(
            "can_local_loss_weight", [0.25, 0.5, 1.0, 1.5]
        ),
        can_margin_loss_weight=margin_weight,
        can_margin_target=(
            trial.suggest_categorical("can_margin_target", [0.3, 0.5, 0.75, 1.0])
            if margin_weight > 0
            else 0.3
        ),
    )


def _dataset_config(
    nas: FewShotNASConfig,
    spec: FewShotArchitectureSpec,
    *,
    sequence_length: int,
    seed: int,
) -> PainDatasetConfig:
    return PainDatasetConfig(
        seed=seed,
        deterministic_ops=False,
        dataset_source="biovid_part_a",
        modalities=nas.modalities,
        task_class_ids=nas.raw_class_ids,
        sequence_length=sequence_length,
        k_shot=nas.k_shot,
        q_query=nas.q_query,
        task_normalize_mode="split",
        task_construction_mode="single_subject",
        attention_mode="can",
        can_support_mode="learned_prototype_memory",
        learned_prototype_slots_per_class=spec.learned_prototype_slots_per_class,
        prototype_bank_init_samples_per_class=spec.prototype_bank_init_samples_per_class,
        prototype_finetune_epochs=1,
        prototype_finetune_tasks_per_epoch=None,
        source_subject_prototype_vote_enabled=False,
        matched_query_eval=False,
        k_shot_adaptation_steps=0,
        train_batch_size=nas.task_batch_size,
        task_chunk_size=nas.task_batch_size,
        tasks_per_epoch=nas.tasks_per_epoch,
        num_epochs=nas.max_epochs,
        disable_validation=True,
        disable_training_logging=True,
        train_prefetch_batches=2,
        encoder_backend="crossmod",
        # The per-modality EEGNet frontend is deliberately fixed.
        eegnet_temporal_filters=8,
        eegnet_depth_multiplier=2,
        eegnet_separable_filters=16,
        eegnet_temporal_kernel_size=64,
        eegnet_separable_kernel_size=16,
        eegnet_pool_size_1=4,
        eegnet_pool_size_2=8,
        eegnet_dropout_rate=0.25,
        eegnet_l2_weight=1e-4,
        eegnet_normalization="group",
        eegnet_group_norm_groups=4,
        crossmod_num_heads=spec.crossmod_num_heads,
        crossmod_hidden_dim=spec.crossmod_hidden_dim,
        crossmod_num_layers=spec.crossmod_num_layers,
        crossmod_attention_dropout_rate=spec.crossmod_attention_dropout_rate,
        crossmod_ff_activation=spec.crossmod_ff_activation,
        crossmod_fusion_mode=spec.crossmod_fusion_mode,
        can_meta_depth=spec.can_meta_depth,
        can_meta_hidden_dim=spec.can_meta_hidden_dim,
        can_meta_activation=spec.can_meta_activation,
        can_temporal_pooling=spec.can_temporal_pooling,
        can_attention_temperature=spec.can_attention_temperature,
        can_local_pool_temperature=spec.can_local_pool_temperature,
        prototype_feature_normalization=spec.prototype_feature_normalization,
        prototype_aggregation=spec.prototype_aggregation,
        prototype_attention_temperature=spec.prototype_attention_temperature,
        can_logit_scale_initial=spec.can_logit_scale_initial,
        can_local_loss_weight=spec.can_local_loss_weight,
        can_margin_loss_weight=spec.can_margin_loss_weight,
        can_margin_target=spec.can_margin_target,
        lr_schedule="cosine",
        lr_decay_alpha=spec.lr_decay_alpha,
        gaussian_noise_std=0.01,
        enable_window_shift_augmentation=False,
        logging_verbosity=0,
    )


def _load_dataset(data_dir: str, nas: FewShotNASConfig) -> PainMetaDataset:
    base = _dataset_config(
        nas, FewShotArchitectureSpec(), sequence_length=1152, seed=nas.seed
    )
    dataset = PainMetaDataset(
        data_dir=data_dir, config=base, normalize=True, normalize_per_subject=True
    )
    if len(dataset.unique_subjects) != nas.expected_subjects:
        raise ValueError(
            f"Expected {nas.expected_subjects} BioVid subjects, got {len(dataset.unique_subjects)}"
        )
    return dataset


def _make_train_sampler(learner: FewShotPainLearner, split: SubjectSplit, seed: int):
    stats = learner.dataset.compute_split_normalization_stats(
        split.train_subjects, split="train"
    )
    stats = dict(stats)
    stats["subject_ids"] = tuple(split.train_subjects)
    stats["split"] = "train"
    sampler = SixWayKShotSampler(
        dataset=learner.dataset,
        mode="train",
        train_subjects=list(split.train_subjects),
        test_subject=None,
        seed=seed,
        data_split="train",
        normalization_stats=stats,
    )
    return sampler, stats


def _validation_metrics(
    learner: FewShotPainLearner,
    split: SubjectSplit,
    normalization_stats: dict,
    nas: FewShotNASConfig,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    zero_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    original_mode = learner.model.can_support_mode
    try:
        for subject in split.validation_subjects:
            fixed = learner.dataset.build_fixed_normalized_query_task(
                subject, normalization_stats=normalization_stats, split="test"
            )
            learner.model.can_support_mode = "learned_prototype_memory"
            zero = learner.evaluator.evaluate_task_batch_detailed(
                [fixed], forward_batch_size=1,
                can_support_mode="learned_prototype_memory",
            )[0]
            zero_rows.append(
                {"subject": subject, **zero["metrics"], "loss": zero["loss"]}
            )

            tasks = []
            for repeat in range(nas.support_repeats):
                repeat_seed = int(
                    np.random.SeedSequence([nas.seed, subject, repeat, 7919])
                    .generate_state(1, dtype=np.uint32)[0]
                )
                task = learner.dataset.sample_support_for_fixed_query(
                    subject,
                    k_shot=nas.k_shot,
                    fixed_query_task=fixed,
                    normalization_stats=normalization_stats,
                    rng=np.random.default_rng(repeat_seed),
                    support_split="train",
                    repeat_index=repeat,
                    repeat_seed=repeat_seed,
                )
                if np.intersect1d(task["support_indices"], task["query_indices"]).size:
                    raise RuntimeError("Validation support/query leakage")
                tasks.append(task)
            learner.model.can_support_mode = "sampled"
            results = learner.evaluator.evaluate_task_batch_detailed(
                tasks,
                forward_batch_size=nas.task_batch_size,
                can_support_mode="sampled",
            )
            for task, result in zip(tasks, results):
                repeat_rows.append(
                    {
                        "subject": subject,
                        "repeat": task["repeat_index"],
                        "repeat_seed": task["repeat_seed"],
                        **result["metrics"],
                        "loss": result["loss"],
                    }
                )
    finally:
        learner.model.can_support_mode = original_mode

    zero_by_subject = {row["subject"]: float(row["accuracy"]) for row in zero_rows}
    k_by_subject = {
        subject: float(np.mean([r["accuracy"] for r in repeat_rows if r["subject"] == subject]))
        for subject in split.validation_subjects
    }
    zero_mean = float(np.mean(list(zero_by_subject.values())))
    k_mean = float(np.mean(list(k_by_subject.values())))
    return {
        "zero_shot_accuracy": zero_mean,
        "k_shot_accuracy": k_mean,
        "objective": 0.5 * (zero_mean + k_mean),
    }, zero_rows, repeat_rows


def _train_epoch(learner: FewShotPainLearner, sampler, tasks_per_epoch: int) -> dict:
    losses, accuracies = [], []
    for _, arrays in learner.task_pipeline.iter_prefetched_task_batches(
        sampler, tasks_per_epoch
    ):
        tensors = (
            tf.convert_to_tensor(arrays[0], tf.float32),
            tf.convert_to_tensor(arrays[1], tf.int32),
            tf.convert_to_tensor(arrays[2], tf.float32),
            tf.convert_to_tensor(arrays[3], tf.int32),
        )
        loss, _, accuracy, _, _ = learner.engine.train_batch_step_tensors(*tensors)
        losses.append(float(loss))
        accuracies.append(float(accuracy))
    return {"train_loss": float(np.mean(losses)), "train_accuracy": float(np.mean(accuracies))}


def _finetune_prototype_memory(learner: FewShotPainLearner, sampler) -> None:
    """Run the standard phase-2 bank update without discarding phase-1 state."""
    prototype_epochs = max(0, int(learner.config.prototype_finetune_epochs))
    if prototype_epochs == 0:
        return
    updates_per_epoch = learner._resolve_prototype_finetune_tasks_per_epoch(sampler)
    phase1_optimizer = learner.engine.optimizer
    try:
        learner.engine.restart_optimizer_for_prototype_phase(
            updates_per_epoch=updates_per_epoch,
            num_epochs=prototype_epochs,
        )
        for _ in range(prototype_epochs):
            for _, arrays in learner._iter_prototype_finetune_task_batches(
                sampler, updates_per_epoch
            ):
                learner._train_prototype_memory_batch_step_tensors(
                    support_x_batch=tf.convert_to_tensor(arrays[0], tf.float32),
                    support_y_batch=tf.convert_to_tensor(arrays[1], tf.int32),
                    query_x_batch=tf.convert_to_tensor(arrays[2], tf.float32),
                    query_y_batch=tf.convert_to_tensor(arrays[3], tf.int32),
                )
    finally:
        learner.engine.optimizer = phase1_optimizer


def _fit(
    dataset: PainMetaDataset,
    split: SubjectSplit,
    nas: FewShotNASConfig,
    spec: FewShotArchitectureSpec,
    *,
    epochs: int,
    seed: int,
    trial: optuna.Trial | None,
) -> tuple[FewShotPainLearner, dict[str, Any]]:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    config = _dataset_config(
        nas, spec, sequence_length=int(dataset.X.shape[1]), seed=seed
    )
    learner = FewShotPainLearner(
        config=config, data_dir=str(dataset.data_dir),
        learning_rate=spec.learning_rate, dataset=dataset,
    )
    parameter_count = int(
        sum(np.prod(tuple(int(dim) for dim in weight.shape)) for weight in learner.model.weights)
    )
    if parameter_count > nas.max_parameters:
        learner.engine.release_model_resources()
        raise optuna.TrialPruned(
            f"Model has {parameter_count:,} parameters; limit is {nas.max_parameters:,}"
        )
    sampler, stats = _make_train_sampler(learner, split, seed)
    history, best = [], None
    best_weights = None
    stale_epochs = 0
    sentinel_subject = int(max(dataset.unique_subjects)) + 1
    for epoch in range(epochs):
        row = {"epoch": epoch + 1, **_train_epoch(learner, sampler, nas.tasks_per_epoch)}
        learner._initialize_prototype_bank_from_training_samples(
            fold=0, test_subject=sentinel_subject, train_sampler=sampler
        )
        _finetune_prototype_memory(learner, sampler)
        metrics, _, _ = _validation_metrics(learner, split, stats, nas)
        row.update(metrics)
        history.append(row)
        if trial is None or best is None or metrics["objective"] > best["objective"]:
            best = dict(row)
            best_weights = learner.model.get_weights()
            stale_epochs = 0
        else:
            stale_epochs += 1
        if trial is not None:
            trial.report(metrics["objective"], step=epoch)
            if trial.should_prune():
                learner.engine.release_model_resources()
                raise optuna.TrialPruned(f"Pruned after epoch {epoch + 1}")
        if stale_epochs > nas.search_patience:
            break
    assert best is not None and best_weights is not None
    learner.model.set_weights(best_weights)
    final_metrics, zero_rows, repeat_rows = _validation_metrics(
        learner, split, stats, nas
    )
    return learner, {
        "parameter_count": parameter_count,
        "best_epoch": int(best["epoch"]),
        "metrics": final_metrics,
        "history": history,
        "zero_shot_subject_metrics": zero_rows,
        "k_shot_repeat_metrics": repeat_rows,
    }


def _trial_rows(study: optuna.Study) -> list[dict[str, Any]]:
    rows = []
    for trial in study.trials:
        row = {"trial_number": trial.number, "state": trial.state.name, "value": trial.value}
        row.update({f"param_{k}": v for k, v in trial.params.items()})
        row.update({f"attr_{k}": v for k, v in trial.user_attrs.items()})
        rows.append(row)
    return rows


def run_search(
    data_dir: str,
    config: FewShotNASConfig,
    output_dir: Path,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _load_dataset(data_dir, config)
    split = deterministic_subject_split(
        dataset.unique_subjects,
        train_count=config.train_subject_count,
        validation_count=config.validation_subject_count,
        seed=config.seed,
    )
    manifest = {
        "stage": "search", "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "train_subjects": split.train_subjects,
        "validation_subjects": split.validation_subjects,
        "subject_keys": getattr(dataset, "predefined_subject_int_to_key", {}),
    }
    ensure_manifest(output_dir / "manifest.json", manifest, resume=resume)
    storage = f"sqlite:///{(output_dir / 'study.sqlite3').resolve()}"
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=storage, direction="maximize",
        load_if_exists=resume,
        sampler=optuna.samplers.TPESampler(
            seed=config.seed, n_startup_trials=20, multivariate=True, group=True
        ),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=config.max_epochs, reduction_factor=3
        ),
    )
    if not study.trials:
        study.enqueue_trial(FewShotArchitectureSpec().to_dict())

    trials_dir = output_dir / "trials"
    trials_dir.mkdir(exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        spec = suggest_architecture(trial)
        started = time.perf_counter()
        trial_dir = trials_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(exist_ok=True)
        learner = None
        try:
            learner, result = _fit(
                dataset, split, config, spec,
                epochs=config.max_epochs,
                seed=config.seed + trial.number,
                trial=trial,
            )
            trial.set_user_attr("best_epoch", result["best_epoch"])
            trial.set_user_attr("parameter_count", result["parameter_count"])
            trial.set_user_attr("zero_shot_accuracy", result["metrics"]["zero_shot_accuracy"])
            trial.set_user_attr("k_shot_accuracy", result["metrics"]["k_shot_accuracy"])
            trial.set_user_attr("elapsed_seconds", time.perf_counter() - started)
            atomic_write_json(trial_dir / "result.json", {"architecture": spec.to_dict(), **result})
            return float(result["metrics"]["objective"])
        except (tf.errors.ResourceExhaustedError, MemoryError) as exc:
            raise optuna.TrialPruned(f"Resource limit: {exc}") from exc
        finally:
            if learner is not None:
                learner.engine.release_model_resources()
            gc.collect()

    def persist(study_obj: optuna.Study, _trial=None) -> None:
        atomic_write_csv(output_dir / "trials.csv", _trial_rows(study_obj))
        complete = [t for t in study_obj.trials if t.value is not None and t.state.name == "COMPLETE"]
        if complete:
            best = max(complete, key=lambda item: float(item.value))
            atomic_write_json(
                output_dir / "best_architecture.json",
                {
                    "architecture": architecture_from_parameters(best.params).to_dict(),
                    "best_trial_number": best.number,
                    "best_objective": best.value,
                    "best_epoch": best.user_attrs["best_epoch"],
                    "zero_shot_accuracy": best.user_attrs["zero_shot_accuracy"],
                    "k_shot_accuracy": best.user_attrs["k_shot_accuracy"],
                    "parameter_count": best.user_attrs["parameter_count"],
                    "train_subjects": split.train_subjects,
                    "validation_subjects": split.validation_subjects,
                    "protocol": "tuning-only matched Train-support/Test-query validation",
                },
            )

    finished_trials = sum(trial.state.is_finished() for trial in study.trials)
    remaining = max(0, config.n_trials - finished_trials)
    if remaining:
        study.optimize(
            objective, n_trials=remaining, callbacks=[persist],
            gc_after_trial=True, show_progress_bar=False,
        )
    persist(study)
    return {
        "trial_count": sum(t.state.is_finished() for t in study.trials),
        "study_record_count": len(study.trials),
        "completed_trials": sum(t.state.name == "COMPLETE" for t in study.trials),
        "best_architecture": str(output_dir / "best_architecture.json"),
    }


def run_refit(
    data_dir: str,
    config: FewShotNASConfig,
    search_dir: Path,
    output_dir: Path,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = read_json(Path(search_dir) / "best_architecture.json")
    spec = FewShotArchitectureSpec.from_dict(best["architecture"])
    manifest = {
        "stage": "refit",
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "architecture": spec.to_dict(),
        "selected_epoch": int(best["best_epoch"]),
        "train_subjects": best["train_subjects"],
        "validation_subjects": best["validation_subjects"],
    }
    ensure_manifest(output_dir / "manifest.json", manifest, resume=resume)
    result_path = output_dir / "result.json"
    if resume and result_path.exists():
        return read_json(result_path)
    dataset = _load_dataset(data_dir, config)
    split = deterministic_subject_split(
        dataset.unique_subjects,
        train_count=config.train_subject_count,
        validation_count=config.validation_subject_count,
        seed=config.seed,
    )
    if (
        list(split.train_subjects) != list(best["train_subjects"])
        or list(split.validation_subjects) != list(best["validation_subjects"])
    ):
        raise ValueError("Refit subject split does not match the search split")
    learner = None
    try:
        learner, result = _fit(
            dataset, split, config, spec,
            epochs=int(best["best_epoch"]),
            seed=config.seed + 1_000_000,
            trial=None,
        )
        learner.model.save_weights(output_dir / "model.weights.h5")
        atomic_write_csv(
            output_dir / "zero_shot_subject_metrics.csv",
            result["zero_shot_subject_metrics"],
        )
        atomic_write_csv(
            output_dir / "k_shot_repeat_metrics.csv",
            result["k_shot_repeat_metrics"],
        )
        payload = {
            "architecture": spec.to_dict(), "selected_epoch": best["best_epoch"],
            "metrics": result["metrics"], "parameter_count": result["parameter_count"],
            "train_subjects": split.train_subjects,
            "validation_subjects": split.validation_subjects,
            "protocol": "tuning-only matched Train-support/Test-query validation",
            "model_weights": str(output_dir / "model.weights.h5"),
        }
        atomic_write_json(result_path, payload)
        return payload
    finally:
        if learner is not None:
            learner.engine.release_model_resources()


def run_all(
    data_dir: str,
    config: FewShotNASConfig,
    output_dir: Path,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    search = run_search(data_dir, config, output_dir / "search", resume=resume)
    refit = run_refit(
        data_dir, config, output_dir / "search", output_dir / "refit", resume=resume
    )
    return {"search": search, "refit": refit, "output_dir": str(output_dir)}
