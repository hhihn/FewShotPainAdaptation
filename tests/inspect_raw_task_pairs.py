import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loaders.loso_cross_validator import LOSOCrossValidator
from data_loaders.pain_ds_config import PainDatasetConfig
from data_loaders.pain_meta_dataset import PainMetaDataset
from utils.logger import setup_logger
from utils.reproducibility import set_global_reproducibility


def _sample_pairwise_train_val_tasks(
    dataset: PainMetaDataset,
    support_subject: int,
    query_subject: int,
    k_shot: int,
    q_query: int,
    normalize_mode: str,
    seed: int,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    support_indices_by_class = []
    query_indices_by_class = []
    for class_id in range(dataset.config.n_way):
        support_indices = dataset.index[int(support_subject)][class_id]
        query_indices = dataset.index[int(query_subject)][class_id]
        if len(support_indices) < 2 * k_shot:
            raise ValueError(
                f"Support subject {support_subject}, class {class_id} does not have "
                f"enough samples for disjoint train/val support splits with k_shot={k_shot}."
            )
        if len(query_indices) < 2 * q_query:
            raise ValueError(
                f"Query subject {query_subject}, class {class_id} does not have "
                f"enough samples for disjoint train/val query splits with q_query={q_query}."
            )
        sampled_support = rng.choice(support_indices, size=2 * k_shot, replace=False)
        sampled_query = rng.choice(query_indices, size=2 * q_query, replace=False)
        support_indices_by_class.append(sampled_support)
        query_indices_by_class.append(sampled_query)

    def build_task(offset: int) -> dict:
        support_X, support_y = [], []
        query_X, query_y = [], []
        support_subjects, query_subjects = [], []
        for class_id in range(dataset.config.n_way):
            support_idx = support_indices_by_class[class_id][
                offset * k_shot : (offset + 1) * k_shot
            ]
            query_idx = query_indices_by_class[class_id][
                offset * q_query : (offset + 1) * q_query
            ]
            support_X.append(dataset.X[support_idx])
            query_X.append(dataset.X[query_idx])
            support_y.append(np.full(k_shot, class_id, dtype=np.int32))
            query_y.append(np.full(q_query, class_id, dtype=np.int32))
            support_subjects.append(np.full(k_shot, support_subject, dtype=np.int32))
            query_subjects.append(np.full(q_query, query_subject, dtype=np.int32))

        support_X = np.concatenate(support_X, axis=0)
        query_X = np.concatenate(query_X, axis=0)
        support_y = np.concatenate(support_y, axis=0)
        query_y = np.concatenate(query_y, axis=0)
        support_subjects = np.concatenate(support_subjects, axis=0)
        query_subjects = np.concatenate(query_subjects, axis=0)

        if normalize_mode == "subject":
            support_X = dataset._normalize_data_by_subjects(support_X, support_subjects)
            query_X = dataset._normalize_data_by_subjects(query_X, query_subjects)
        elif normalize_mode == "support":
            stats = dataset._compute_batch_stats(support_X)
            support_X = dataset._apply_stats(support_X, stats)
            query_X = dataset._apply_stats(query_X, stats)
        elif normalize_mode == "none":
            pass
        else:
            raise ValueError(f"Unsupported normalize_mode={normalize_mode}")

        return {
            "support_X": support_X,
            "support_y": support_y,
            "query_X": query_X,
            "query_y": query_y,
            "support_subjects": support_subjects,
            "query_subjects": query_subjects,
        }

    return build_task(offset=0), build_task(offset=1)


def _flatten_features(task: dict) -> tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([task["support_X"], task["query_X"]], axis=0)
    y = np.concatenate([task["support_y"], task["query_y"]], axis=0)
    return X.reshape(X.shape[0], -1), y


def _summary_features(task: dict) -> tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([task["support_X"], task["query_X"]], axis=0)
    y = np.concatenate([task["support_y"], task["query_y"]], axis=0)
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    minimum = np.min(X, axis=1)
    maximum = np.max(X, axis=1)
    rms = np.sqrt(np.mean(np.square(X), axis=1))
    diff_energy = np.mean(np.square(np.diff(X, axis=1)), axis=1)
    features = np.concatenate([mean, std, minimum, maximum, rms, diff_energy], axis=1)
    return features, y


def _modality_summary_stats(
    X: np.ndarray,
    y: np.ndarray,
    modality_names: tuple[str, ...],
) -> dict:
    summaries = {}
    for modality_idx, modality_name in enumerate(modality_names):
        modality_values = X[:, :, modality_idx]
        modality_summary = {
            "overall": {
                "mean": float(np.mean(modality_values)),
                "std": float(np.std(modality_values)),
                "min": float(np.min(modality_values)),
                "max": float(np.max(modality_values)),
                "rms": float(np.sqrt(np.mean(np.square(modality_values)))),
                "diff_energy": float(
                    np.mean(np.square(np.diff(modality_values, axis=1)))
                ),
            },
            "by_class": {},
        }
        for class_id in np.unique(y):
            class_values = modality_values[y == class_id]
            modality_summary["by_class"][int(class_id)] = {
                "mean": float(np.mean(class_values)),
                "std": float(np.std(class_values)),
                "min": float(np.min(class_values)),
                "max": float(np.max(class_values)),
                "rms": float(np.sqrt(np.mean(np.square(class_values)))),
                "diff_energy": float(
                    np.mean(np.square(np.diff(class_values, axis=1)))
                ),
            }
        summaries[modality_name] = modality_summary
    return summaries


def _task_modality_comparison(task: dict, modality_names: tuple[str, ...]) -> dict:
    return {
        "support": _modality_summary_stats(
            X=task["support_X"],
            y=task["support_y"],
            modality_names=modality_names,
        ),
        "query": _modality_summary_stats(
            X=task["query_X"],
            y=task["query_y"],
            modality_names=modality_names,
        ),
    }


def _fit_baselines(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
) -> dict[str, float]:
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_X, train_y)
    knn_pred = knn.predict(val_X)

    logreg = LogisticRegression(max_iter=5000, solver="lbfgs")
    logreg.fit(train_X, train_y)
    logreg_pred = logreg.predict(val_X)

    return {
        "knn_acc": float(accuracy_score(val_y, knn_pred)),
        "knn_bal_acc": float(balanced_accuracy_score(val_y, knn_pred)),
        "logreg_acc": float(accuracy_score(val_y, logreg_pred)),
        "logreg_bal_acc": float(balanced_accuracy_score(val_y, logreg_pred)),
    }


def _plot_waveforms(
    train_task: dict,
    val_task: dict,
    modality_names: tuple[str, ...],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(modality_names), 2, figsize=(12, 3.5 * len(modality_names)), sharex=True
    )
    if len(modality_names) == 1:
        axes = np.array([axes])

    for modality_idx, modality_name in enumerate(modality_names):
        train_axis = axes[modality_idx, 0]
        val_axis = axes[modality_idx, 1]

        for class_id in np.unique(train_task["support_y"]):
            support_mask = train_task["support_y"] == class_id
            query_mask = train_task["query_y"] == class_id
            train_axis.plot(
                train_task["support_X"][support_mask][0, :, modality_idx],
                alpha=0.8,
                label=f"class {class_id} support",
            )
            train_axis.plot(
                train_task["query_X"][query_mask][0, :, modality_idx],
                alpha=0.8,
                linestyle="--",
                label=f"class {class_id} query",
            )

            val_support_mask = val_task["support_y"] == class_id
            val_query_mask = val_task["query_y"] == class_id
            val_axis.plot(
                val_task["support_X"][val_support_mask][0, :, modality_idx],
                alpha=0.8,
                label=f"class {class_id} support",
            )
            val_axis.plot(
                val_task["query_X"][val_query_mask][0, :, modality_idx],
                alpha=0.8,
                linestyle="--",
                label=f"class {class_id} query",
            )

        train_axis.set_title(f"Train Task | {modality_name}")
        val_axis.set_title(f"Val Task | {modality_name}")
        train_axis.legend(fontsize=8, loc="upper right")
        val_axis.legend(fontsize=8, loc="upper right")

    figure.suptitle("Raw waveform comparison for one subject pair")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_pca(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    output_path: Path,
) -> None:
    pca = PCA(n_components=2)
    combined = np.concatenate([train_features, val_features], axis=0)
    projected = pca.fit_transform(combined)
    train_projected = projected[: len(train_features)]
    val_projected = projected[len(train_features) :]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for class_id in np.unique(train_labels):
        mask = train_labels == class_id
        axes[0].scatter(
            train_projected[mask, 0],
            train_projected[mask, 1],
            label=f"class {class_id}",
            alpha=0.8,
        )
    axes[0].set_title("Train Task Summary Features (PCA)")
    axes[0].legend()

    for class_id in np.unique(val_labels):
        mask = val_labels == class_id
        axes[1].scatter(
            val_projected[mask, 0],
            val_projected[mask, 1],
            label=f"class {class_id}",
            alpha=0.8,
        )
    axes[1].set_title("Val Task Summary Features (PCA)")
    axes[1].legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect raw train/val task pairs with waveform plots and simple baselines."
    )
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--held-out-subject", type=int, default=None)
    parser.add_argument("--support-subject", type=int, default=None)
    parser.add_argument("--query-subject", type=int, default=None)
    parser.add_argument("--k-shot", type=int, default=2)
    parser.add_argument("--q-query", type=int, default=2)
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="subject",
        choices=("subject", "support", "none"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tests/outputs/diagnostics/raw_task_pairs",
    )
    args = parser.parse_args()

    logger = setup_logger("inspect_raw_task_pairs")
    config = PainDatasetConfig(
        seed=args.seed,
        deterministic_ops=True,
        k_shot=args.k_shot,
        q_query=args.q_query,
        single_loso_fold=True,
    )
    set_global_reproducibility(
        seed=config.seed,
        deterministic_ops=config.deterministic_ops,
        logger=logger,
    )

    dataset = PainMetaDataset(
        data_dir=args.data_dir,
        config=config,
        normalize=True,
        normalize_per_subject=True,
    )
    cv = LOSOCrossValidator(dataset=dataset, seed=config.seed)
    held_out_subject = (
        int(args.held_out_subject)
        if args.held_out_subject is not None
        else int(cv.subjects[0])
    )
    fold = cv.get_fold(held_out_subject)
    train_subjects = [int(subject) for subject in fold["train_subjects"]]
    support_subject = (
        int(args.support_subject)
        if args.support_subject is not None
        else int(train_subjects[0])
    )
    query_subject = (
        int(args.query_subject)
        if args.query_subject is not None
        else int(train_subjects[1])
    )
    if support_subject == query_subject:
        raise ValueError("support_subject and query_subject must differ")

    train_task, val_task = _sample_pairwise_train_val_tasks(
        dataset=dataset,
        support_subject=support_subject,
        query_subject=query_subject,
        k_shot=args.k_shot,
        q_query=args.q_query,
        normalize_mode=args.normalize_mode,
        seed=args.seed,
    )

    raw_train_X, raw_train_y = _flatten_features(train_task)
    raw_val_X, raw_val_y = _flatten_features(val_task)
    summary_train_X, summary_train_y = _summary_features(train_task)
    summary_val_X, summary_val_y = _summary_features(val_task)

    raw_metrics = _fit_baselines(raw_train_X, raw_train_y, raw_val_X, raw_val_y)
    summary_metrics = _fit_baselines(
        summary_train_X, summary_train_y, summary_val_X, summary_val_y
    )
    modality_stats = {
        "train": _task_modality_comparison(train_task, config.modality_names),
        "val": _task_modality_comparison(val_task, config.modality_names),
    }

    output_dir = Path(args.output_dir)
    waveform_path = output_dir / "raw_waveforms.png"
    pca_path = output_dir / "summary_features_pca.png"
    metrics_path = output_dir / "baseline_metrics.json"

    _plot_waveforms(
        train_task=train_task,
        val_task=val_task,
        modality_names=config.modality_names,
        output_path=waveform_path,
    )
    _plot_pca(
        train_features=summary_train_X,
        train_labels=summary_train_y,
        val_features=summary_val_X,
        val_labels=summary_val_y,
        output_path=pca_path,
    )

    metrics = {
        "support_subject": support_subject,
        "query_subject": query_subject,
        "normalize_mode": args.normalize_mode,
        "raw_baselines": raw_metrics,
        "summary_feature_baselines": summary_metrics,
        "modality_stats": modality_stats,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info(f"Saved waveform plot to {waveform_path}")
    logger.info(f"Saved PCA plot to {pca_path}")
    logger.info(f"Saved baseline metrics to {metrics_path}")
    logger.info(f"Raw baselines: {json.dumps(raw_metrics)}")
    logger.info(f"Summary baselines: {json.dumps(summary_metrics)}")
    for split_name in ("train", "val"):
        for subset_name in ("support", "query"):
            logger.info(
                f"{split_name.capitalize()} {subset_name} modality stats: "
                f"{json.dumps(modality_stats[split_name][subset_name])}"
            )


if __name__ == "__main__":
    main()
