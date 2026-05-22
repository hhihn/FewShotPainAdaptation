import argparse
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture.multimodal_proto_net import MultimodalPrototypicalNetwork
from data_loaders.loso_cross_validator import LOSOCrossValidator
from data_loaders.pain_ds_config import PainDatasetConfig
from data_loaders.pain_meta_dataset import PainMetaDataset
from utils.logger import setup_logger
from utils.reproducibility import set_global_reproducibility


def _build_model(config: PainDatasetConfig) -> MultimodalPrototypicalNetwork:
    return MultimodalPrototypicalNetwork(
        sequence_length=config.sequence_length,
        num_sensors=len(config.sensor_idx),
        num_classes=config.n_way,
        embedding_dim=config.embedding_dim,
        distance_metric="cosine",
        classifier_mode=config.classifier_mode,
        eegnet_temporal_filters=config.eegnet_temporal_filters,
        eegnet_depth_multiplier=config.eegnet_depth_multiplier,
        eegnet_separable_filters=config.eegnet_separable_filters,
        eegnet_temporal_kernel_size=config.eegnet_temporal_kernel_size,
        eegnet_separable_kernel_size=config.eegnet_separable_kernel_size,
        eegnet_pool_size_1=config.eegnet_pool_size_1,
        eegnet_pool_size_2=config.eegnet_pool_size_2,
        eegnet_dropout_rate=config.eegnet_dropout_rate,
        eegnet_l2_weight=config.eegnet_l2_weight,
    )


def _build_and_maybe_load_weights(
    model: MultimodalPrototypicalNetwork,
    config: PainDatasetConfig,
    weights_path: str | None,
) -> None:
    dummy_x = tf.zeros(
        (1, config.sequence_length, len(config.sensor_idx)), dtype=tf.float32
    )
    _ = model.encode(dummy_x, training=False)
    if weights_path:
        model.load_weights(weights_path)


def _extract_split_arrays(
    dataset: PainMetaDataset,
    train_subjects: list[int],
    held_out_subject: int,
    normalize_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_raw_labels = np.array(dataset.task_class_ids, dtype=np.int32)
    label_to_episode = {
        raw_label: idx for idx, raw_label in enumerate(valid_raw_labels)
    }
    valid_mask = np.isin(dataset.y, valid_raw_labels)

    X = dataset.X[valid_mask].copy()
    raw_y = dataset.y[valid_mask].astype(np.int32)
    y = np.array([label_to_episode[int(label)] for label in raw_y], dtype=np.int32)
    subjects = dataset.subjects[valid_mask].astype(np.int32)

    if normalize_mode == "subject":
        X = dataset._normalize_data_by_subjects(X, subjects)
    elif normalize_mode == "none":
        pass
    else:
        raise ValueError(
            "normalize_mode must be 'subject' or 'none' for embedding probes"
        )

    train_mask = np.isin(subjects, np.array(train_subjects, dtype=np.int32))
    heldout_mask = subjects == int(held_out_subject)
    return (
        X[train_mask],
        y[train_mask],
        subjects[train_mask],
        X[heldout_mask],
        y[heldout_mask],
        subjects[heldout_mask],
    )


def _encode_in_batches(
    model: MultimodalPrototypicalNetwork,
    X: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(X), batch_size):
        batch_x = tf.constant(X[start : start + batch_size], dtype=tf.float32)
        batch_z = model.encode(batch_x, training=False)
        outputs.append(batch_z.numpy())
    return np.concatenate(outputs, axis=0)


def _fit_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    multi_class: str,
) -> dict[str, float]:
    clf = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        multi_class=multi_class,
        n_jobs=None,
    )
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    return {
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "train_balanced_accuracy": float(balanced_accuracy_score(y_train, train_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe frozen embeddings for pain class and subject information."
    )
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--held-out-subject", type=int, default=None)
    parser.add_argument("--classifier-mode", type=str, default="prototype")
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="subject",
        choices=("subject", "none"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--eegnet-temporal-filters", type=int, default=8)
    parser.add_argument("--eegnet-depth-multiplier", type=int, default=2)
    parser.add_argument("--eegnet-separable-filters", type=int, default=16)
    parser.add_argument("--eegnet-temporal-kernel-size", type=int, default=64)
    parser.add_argument("--eegnet-separable-kernel-size", type=int, default=16)
    parser.add_argument("--eegnet-pool-size-1", type=int, default=4)
    parser.add_argument("--eegnet-pool-size-2", type=int, default=8)
    parser.add_argument("--eegnet-dropout-rate", type=float, default=0.25)
    parser.add_argument("--eegnet-l2-weight", type=float, default=1e-4)
    args = parser.parse_args()

    logger = setup_logger("embedding_probe")
    config = PainDatasetConfig(
        seed=args.seed,
        deterministic_ops=True,
        single_loso_fold=True,
        classifier_mode=args.classifier_mode,
        embedding_dim=args.embedding_dim,
        eegnet_temporal_filters=args.eegnet_temporal_filters,
        eegnet_depth_multiplier=args.eegnet_depth_multiplier,
        eegnet_separable_filters=args.eegnet_separable_filters,
        eegnet_temporal_kernel_size=args.eegnet_temporal_kernel_size,
        eegnet_separable_kernel_size=args.eegnet_separable_kernel_size,
        eegnet_pool_size_1=args.eegnet_pool_size_1,
        eegnet_pool_size_2=args.eegnet_pool_size_2,
        eegnet_dropout_rate=args.eegnet_dropout_rate,
        eegnet_l2_weight=args.eegnet_l2_weight,
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

    model = _build_model(config=config)
    _build_and_maybe_load_weights(model, config=config, weights_path=args.weights_path)

    (
        train_X,
        train_y,
        train_subjects_per_sample,
        heldout_X,
        heldout_y,
        heldout_subjects_per_sample,
    ) = _extract_split_arrays(
        dataset=dataset,
        train_subjects=train_subjects,
        held_out_subject=held_out_subject,
        normalize_mode=args.normalize_mode,
    )

    logger.info(
        f"Embedding probe split: held_out_subject={held_out_subject}, "
        f"train_samples={len(train_X)}, heldout_samples={len(heldout_X)}, "
        f"normalize_mode={args.normalize_mode}, weights_path={args.weights_path}"
    )

    train_embeddings = _encode_in_batches(model, train_X, batch_size=args.batch_size)
    heldout_embeddings = _encode_in_batches(
        model, heldout_X, batch_size=args.batch_size
    )

    class_probe_metrics = _fit_logreg(
        X_train=train_embeddings,
        y_train=train_y,
        X_test=heldout_embeddings,
        y_test=heldout_y,
        multi_class="auto",
    )

    subj_X_train, subj_X_test, subj_y_train, subj_y_test = train_test_split(
        train_embeddings,
        train_subjects_per_sample,
        test_size=0.2,
        random_state=args.seed,
        stratify=train_subjects_per_sample,
    )
    subject_probe_metrics = _fit_logreg(
        X_train=subj_X_train,
        y_train=subj_y_train,
        X_test=subj_X_test,
        y_test=subj_y_test,
        multi_class="multinomial",
    )

    logger.info(
        "Pain-class probe: "
        f"train_acc={class_probe_metrics['train_accuracy']:.4f}, "
        f"heldout_acc={class_probe_metrics['test_accuracy']:.4f}, "
        f"train_bal_acc={class_probe_metrics['train_balanced_accuracy']:.4f}, "
        f"heldout_bal_acc={class_probe_metrics['test_balanced_accuracy']:.4f}"
    )
    logger.info(
        "Subject-ID probe: "
        f"train_acc={subject_probe_metrics['train_accuracy']:.4f}, "
        f"test_acc={subject_probe_metrics['test_accuracy']:.4f}, "
        f"train_bal_acc={subject_probe_metrics['train_balanced_accuracy']:.4f}, "
        f"test_bal_acc={subject_probe_metrics['test_balanced_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
