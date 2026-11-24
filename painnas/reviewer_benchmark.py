"""Paired BioVid protocol and search comparisons for the Colab entry point."""

from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import pandas as pd
import optuna

from painnas.cross_fitted_loso import (
    resolve_continuation_epochs,
    run_uncertainty_aware_block_search,
)
from painnas.data import (
    build_cross_fitted_subject_plan,
    compute_source_normalization,
    indices_for_subjects,
    load_biovid_binary,
    make_tf_dataset,
)
from painnas.flops import estimate_early_fusion_compute
from painnas.io import atomic_write_csv, atomic_write_json, ensure_manifest, read_json
from painnas.loso import classification_metrics
from painnas.model import (
    ArchitectureSpec,
    architecture_from_dict,
    build_model,
    compile_model,
)
from painnas.reviewer_search import PanelConfirmation, ReviewerPolicy, candidate_pool
from painnas.runtime import reset_runtime


def final_fit(
    arrays, config, directory, sources, targets, search, search_dir, seed, resume
):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "result.json"
    if resume and path.exists():
        return read_json(path)
    start = time.monotonic()
    if set(sources).intersection(targets):
        raise ValueError("Final training subjects overlap evaluation targets")
    reset_runtime(seed)
    train = indices_for_subjects(arrays, sources, split_code=arrays.train_split_code)
    test = indices_for_subjects(arrays, targets, split_code=arrays.test_split_code)
    mean, std = compute_source_normalization(arrays.X, train)
    spec = (
        architecture_from_dict(search["architecture"])
        if search
        else ArchitectureSpec.baseline()
    )
    epochs = (
        resolve_continuation_epochs(search, config)
        if search
        else config.search_max_epochs
    )
    model = build_model(
        spec,
        input_shape=(arrays.num_modalities, arrays.sequence_length, 1),
        num_classes=config.num_classes,
    )
    if search:
        model.load_weights(search_dir / search["warm_start_checkpoint"])
    compile_model(model, spec)

    def dataset(idx, training):
        return make_tf_dataset(
            arrays,
            idx,
            mean=mean,
            std=std,
            batch_size=config.batch_size,
            training=training,
            seed=seed,
        )

    history = model.fit(dataset(train, True), epochs=epochs, verbose=0)
    if not np.isfinite(history.history["loss"]).all():
        raise FloatingPointError("Nonfinite final training loss")
    probabilities = model.predict(dataset(test, False), verbose=0)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("Nonfinite target predictions")
    predictions = pd.DataFrame(
        probabilities, columns=[f"prob_raw_{v}" for v in config.raw_class_ids]
    )
    predictions["sample"] = test
    predictions["subject"] = arrays.subjects[test]
    predictions["true_raw"] = np.asarray(config.raw_class_ids)[arrays.y[test]]
    predictions["predicted_raw"] = np.asarray(config.raw_class_ids)[
        probabilities.argmax(axis=1)
    ]
    atomic_write_csv(directory / "predictions.csv", predictions.to_dict("records"))
    metrics = []
    flops = estimate_early_fusion_compute(
        spec,
        sequence_length=arrays.sequence_length,
        modality_count=arrays.num_modalities,
        num_classes=config.num_classes,
    ).flops
    for subject in targets:
        mask = arrays.subjects[test] == subject
        row = classification_metrics(arrays.y[test][mask], probabilities[mask])
        metrics.append(
            dict(
                subject=subject,
                accuracy=row["accuracy"],
                macro_f1=row["macro_f1"],
                parameters=model.count_params(),
                flops=flops,
            )
        )
    result = dict(
        metrics=metrics,
        sources=sources,
        targets=targets,
        epochs=epochs,
        seconds=time.monotonic() - start,
    )
    atomic_write_json(path, result)
    return result


def block_search(
    arrays, cfg, directory, block_index, targets, inner, policy, resume, verbose
):
    time_path = directory / "elapsed.json"
    settings = policy.manifest if policy else None
    if resume and time_path.exists():
        timing = read_json(time_path)
        if timing["policy"] != settings:
            raise ValueError(f"Search configuration changed: {directory}")
        return read_json(directory / "best_architecture.json"), timing["seconds"]

    start = time.monotonic()
    search = run_uncertainty_aware_block_search(
        arrays,
        cfg,
        directory,
        outer_block_index=block_index,
        outer_block_subjects=targets,
        inner_folds=inner,
        resume=resume,
        verbose=verbose,
        reviewer_policy=policy,
    )
    if policy:
        seconds = read_json(directory / "budget.json")["seconds"]
    else:
        study = optuna.load_study(
            study_name=search["study_name"],
            storage=f"sqlite:///{(directory / 'study.sqlite3').resolve()}",
        )
        trial_seconds = sum(
            t.duration.total_seconds() for t in study.trials if t.duration is not None
        )
        seconds = max(time.monotonic() - start, trial_seconds)
    atomic_write_json(time_path, dict(seconds=seconds, policy=settings))
    return search, seconds


def screening_diagnostic(
    arrays, cfg, directory, targets, inner, block_index, seed, size, resume
):
    """Independently score a held-back architecture panel after search is finished."""
    path = directory / "correlations.json"
    if resume and path.exists():
        return
    start = time.monotonic()
    pool = candidate_pool(seed, size)
    scores = {}
    valid = set(range(size))
    for method in ("synflow", "weight_sharing"):
        policy = ReviewerPolicy(method, None, arrays, cfg, inner, seed, size)
        policy.deadline = float("inf")
        eligible, scores[method] = policy.rank_pool(pool)
        valid.intersection_update(eligible)
    rows = []
    for i in sorted(valid):
        search = run_uncertainty_aware_block_search(
            arrays,
            cfg,
            directory / f"candidate_{i}",
            outer_block_index=block_index,
            outer_block_subjects=targets,
            inner_folds=inner,
            resume=resume,
            verbose=0,
            reviewer_policy=PanelConfirmation(pool[i]),
        )
        rows.append(
            dict(
                candidate=i,
                confirmed=search["best_selection_score"],
                **{method: values[i] for method, values in scores.items()},
            )
        )
    atomic_write_csv(directory / "panel.csv", rows)
    ranks = pd.DataFrame(rows).rank()
    correlations = {}
    for method in scores:
        rho = np.nan
        if (
            len(rows) >= 2
            and min(ranks[method].nunique(), ranks["confirmed"].nunique()) > 1
        ):
            rho = ranks[method].corr(ranks["confirmed"])
        correlations[method] = float(rho) if np.isfinite(rho) else None
    atomic_write_json(
        path, dict(panel=pool, spearman=correlations, seconds=time.monotonic() - start)
    )


def run_reviewer_benchmark(
    data_dir,
    base,
    output,
    *,
    seeds=(42, 43, 44),
    split_seed=42,
    tasks=((0, 4), (0, 1, 2, 3, 4)),
    resume=True,
    pool_size=16,
    methods=("tpe", "random", "synflow", "weight_sharing", "generative"),
    budget_seconds=None,
    diagnostic_size=4,
    verbose=1,
):
    """Run both experiments. Budget overrides map class counts to seconds per block."""
    output = Path(output)
    if pool_size < 1 or diagnostic_size < 0 or not seeds or not tasks:
        raise ValueError("Invalid benchmark dimensions")
    if set(methods) - {"random", "tpe", "synflow", "weight_sharing", "generative"}:
        raise ValueError("Unknown reviewer search method")
    rows = []
    if budget_seconds is not None:
        for labels in tasks:
            seconds = budget_seconds[len(labels)]
            if not np.isfinite(seconds) or seconds <= 0:
                raise ValueError("Each task needs a positive, finite per-block budget")
    for labels in tasks:
        task = f"classes_{'_'.join(map(str, labels))}"
        config = replace(
            base,
            raw_class_ids=tuple(labels),
            num_classes=len(labels),
            fusion_mode="early",
            architecture_selection_mode="uncertainty_aware",
            uncertainty_beta=1.0,
        )
        arrays = load_biovid_binary(str(data_dir), config)
        arrays.validate(config, require_expected_subjects=True)
        subjects = tuple(sorted(map(int, arrays.unique_subjects)))
        for seed in seeds:
            config = replace(config, seed=seed)
            root = output / task / f"seed_{seed}"
            for blocks, arms in [
                (config.outer_block_count, ["fixed", "grouped", "painnas", *methods]),
                (len(subjects), ["nested"]),
            ]:
                plan = build_cross_fitted_subject_plan(
                    subjects,
                    outer_block_count=blocks,
                    inner_fold_count=config.inner_fold_count,
                    seed=split_seed,
                )
                run = root / f"blocks_{blocks}"
                cfg = replace(config, outer_block_count=blocks)
                ensure_manifest(
                    run / "manifest.json",
                    dict(
                        config=cfg.to_dict(),
                        split_seed=split_seed,
                        blocks=plan.outer_blocks,
                        inner=plan.inner_folds_by_block,
                        fixed_epochs=cfg.search_max_epochs,
                        diagnostic_size=diagnostic_size,
                    ),
                    resume=resume,
                )
                for b, (targets, inner) in enumerate(
                    zip(plan.outer_blocks, plan.inner_folds_by_block), 1
                ):
                    block = run / f"block_{b:03d}"
                    for arm in arms:
                        if verbose:
                            print(
                                task,
                                "seed",
                                seed,
                                "block",
                                b,
                                "/",
                                blocks,
                                arm,
                                flush=True,
                            )
                        policy = None
                        search_arm = (
                            "reference"
                            if arm in {"grouped", "painnas", "nested"}
                            else arm
                        )
                        search_dir = block / search_arm / "search"
                        search = None
                        seconds = 0.0
                        if arm != "fixed":
                            if arm in methods:
                                reference = read_json(
                                    block / "reference/search/elapsed.json"
                                )
                                budget = (
                                    budget_seconds[len(labels)]
                                    if budget_seconds
                                    else reference["seconds"]
                                )
                                policy = ReviewerPolicy(
                                    arm,
                                    budget,
                                    arrays,
                                    cfg,
                                    inner,
                                    seed + 10000 * b,
                                    pool_size,
                                )
                            search, seconds = block_search(
                                arrays,
                                cfg,
                                search_dir,
                                b,
                                targets,
                                inner,
                                policy,
                                resume or arm == "painnas",
                                verbose,
                            )
                        groups = (
                            [targets] if arm == "grouped" else [(s,) for s in targets]
                        )
                        for group in groups:
                            fit_sources = tuple(sorted(set(subjects) - set(group)))
                            result = final_fit(
                                arrays,
                                cfg,
                                block / arm / f"fit_{group[0]}",
                                fit_sources,
                                group,
                                search,
                                search_dir,
                                seed + subjects.index(group[0]),
                                resume,
                            )
                            for metric in result["metrics"]:
                                rows.append(
                                    dict(
                                        task=task,
                                        seed=seed,
                                        block=b,
                                        arm=arm,
                                        **metric,
                                        search_seconds=seconds / len(targets),
                                        fit_seconds=result["seconds"] / len(group),
                                    )
                                )
                    if diagnostic_size and blocks != len(subjects):
                        screening_diagnostic(
                            arrays,
                            cfg,
                            block / "diagnostic",
                            targets,
                            inner,
                            b,
                            seed + b * 10000 + 500000,
                            diagnostic_size,
                            resume,
                        )
    frame = pd.DataFrame(rows)
    atomic_write_csv(output / "subject_metrics.csv", rows)
    summary = frame.groupby(["task", "seed", "arm"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        macro_f1=("macro_f1", "mean"),
        parameters=("parameters", "mean"),
        flops=("flops", "mean"),
        subjects=("subject", "size"),
        search_seconds=("search_seconds", "sum"),
        fit_seconds=("fit_seconds", "sum"),
    )
    summary["gpu_hours"] = (summary.search_seconds + summary.fit_seconds) / 3600
    atomic_write_csv(output / "summary.csv", summary.to_dict("records"))
    paired = frame.pivot(
        index=["task", "seed", "subject"], columns="arm", values="accuracy"
    )
    differences = {
        f"{a}_minus_{b}_pp": 100 * (paired[a] - paired[b])
        for a, b in [
            ("painnas", "fixed"),
            ("painnas", "grouped"),
            ("painnas", "nested"),
            *((m, "tpe") for m in methods if m != "tpe" and "tpe" in methods),
        ]
    }
    contrasts = pd.DataFrame(differences).reset_index()
    atomic_write_csv(output / "paired_differences.csv", contrasts.to_dict("records"))
    membership = frame.loc[frame.arm == "painnas", ["task", "seed", "subject", "block"]]
    contrasts = contrasts.merge(membership, validate="one_to_one")
    intervals = []
    for (task, seed), group in contrasts.groupby(["task", "seed"]):
        counts = group.groupby("block").size().to_numpy()
        rng = np.random.default_rng(seed)
        sampled = rng.integers(len(counts), size=(base.bootstrap_samples, len(counts)))
        for name in differences:
            totals = group.groupby("block")[name].sum().to_numpy()
            estimates = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
            low, high = np.quantile(estimates, [0.025, 0.975])
            intervals.append(
                dict(
                    task=task,
                    seed=seed,
                    contrast=name,
                    mean_pp=group[name].mean(),
                    block_bootstrap_low_pp=low,
                    block_bootstrap_high_pp=high,
                )
            )
    atomic_write_csv(output / "paired_summary.csv", intervals)
    return summary
