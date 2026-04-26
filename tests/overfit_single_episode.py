import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Allow running this file directly from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture.mulitmodal_proto_net import MultimodalPrototypicalNetwork
from data_loaders.loso_cross_validator import LOSOCrossValidator
from data_loaders.pain_ds_config import PainDatasetConfig
from data_loaders.pain_meta_dataset import PainMetaDataset
from utils.logger import setup_logger
from utils.reproducibility import set_global_reproducibility


@dataclass
class StepMetrics:
    step: int
    loss: float
    accuracy: float
    heldout_val_loss: float | None = None
    heldout_val_accuracy: float | None = None
    within_class_cosine: float | None = None
    between_class_cosine: float | None = None
    cross_bank_same_class_cosine: float | None = None
    cross_bank_diff_class_cosine: float | None = None
    task_query_support_same_class_cosine: float | None = None
    task_query_support_diff_class_cosine: float | None = None
    task_query_prototype_same_class_cosine: float | None = None
    task_query_prototype_diff_class_cosine: float | None = None


def _to_tensors(task: dict) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    return (
        tf.constant(task["support_X"], dtype=tf.float32),
        tf.constant(task["support_y"], dtype=tf.int32),
        tf.constant(task["query_X"], dtype=tf.float32),
        tf.constant(task["query_y"], dtype=tf.int32),
    )


def _accuracy_from_logits(logits: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))


def _mean_or_nan(values: tf.Tensor) -> tf.Tensor:
    if tf.size(values) == 0:
        return tf.constant(float("nan"), dtype=tf.float32)
    return tf.reduce_mean(values)


def _pairwise_bank_cosine_stats(
    embeddings: tf.Tensor, labels: tf.Tensor
) -> tuple[float, float]:
    normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1)
    similarities = tf.matmul(
        normalized_embeddings, normalized_embeddings, transpose_b=True
    )
    labels = tf.reshape(labels, [-1])
    same_class_mask = tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0))
    diag_mask = tf.eye(tf.shape(labels)[0], dtype=tf.bool)
    same_class_mask = tf.logical_and(same_class_mask, tf.logical_not(diag_mask))
    diff_class_mask = tf.logical_and(
        tf.logical_not(same_class_mask), tf.logical_not(diag_mask)
    )

    same_values = tf.boolean_mask(similarities, same_class_mask)
    diff_values = tf.boolean_mask(similarities, diff_class_mask)
    return float(_mean_or_nan(same_values)), float(_mean_or_nan(diff_values))


def _cross_bank_cosine_stats(
    train_embeddings: tf.Tensor,
    train_labels: tf.Tensor,
    val_embeddings: tf.Tensor,
    val_labels: tf.Tensor,
) -> tuple[float, float]:
    train_embeddings = tf.nn.l2_normalize(train_embeddings, axis=1)
    val_embeddings = tf.nn.l2_normalize(val_embeddings, axis=1)
    similarities = tf.matmul(train_embeddings, val_embeddings, transpose_b=True)
    train_labels = tf.reshape(train_labels, [-1])
    val_labels = tf.reshape(val_labels, [-1])
    same_class_mask = tf.equal(
        tf.expand_dims(train_labels, 1), tf.expand_dims(val_labels, 0)
    )
    diff_class_mask = tf.logical_not(same_class_mask)
    same_values = tf.boolean_mask(similarities, same_class_mask)
    diff_values = tf.boolean_mask(similarities, diff_class_mask)
    return float(_mean_or_nan(same_values)), float(_mean_or_nan(diff_values))


def _collect_bank_embeddings(
    model: MultimodalPrototypicalNetwork, fixed_tasks: list[dict]
) -> tuple[tf.Tensor, tf.Tensor]:
    bank_embeddings = []
    bank_labels = []
    for task in fixed_tasks:
        support_x, support_y, query_x, query_y = _to_tensors(task)
        support_embeddings = model.encode(support_x, training=False)
        query_embeddings = model.encode(query_x, training=False)
        bank_embeddings.append(
            tf.concat([support_embeddings, query_embeddings], axis=0)
        )
        bank_labels.append(tf.concat([support_y, query_y], axis=0))
    return tf.concat(bank_embeddings, axis=0), tf.concat(bank_labels, axis=0)


def _task_local_cosine_stats(
    model: MultimodalPrototypicalNetwork,
    fixed_tasks: list[dict],
) -> tuple[float, float, float, float]:
    query_support_same_values = []
    query_support_diff_values = []
    query_prototype_same_values = []
    query_prototype_diff_values = []

    for task in fixed_tasks:
        support_x, support_y, query_x, query_y = _to_tensors(task)
        support_embeddings = model.encode(support_x, training=False)
        query_embeddings = model.encode(query_x, training=False)
        prototypes = model._compute_prototypes(support_embeddings, support_y)

        support_similarities = model.compute_support_to_query_similarity(
            query_embeddings=query_embeddings,
            support_embeddings=support_embeddings,
        )
        prototype_similarities = model.compute_similarity_scores(
            query_embeddings=query_embeddings,
            prototype_embeddings=prototypes,
        )

        query_labels_exp = tf.expand_dims(query_y, 1)
        support_labels_exp = tf.expand_dims(support_y, 0)
        same_support_mask = tf.equal(query_labels_exp, support_labels_exp)
        diff_support_mask = tf.logical_not(same_support_mask)

        class_ids = tf.range(model.num_classes, dtype=query_y.dtype)
        class_ids = tf.reshape(class_ids, [1, -1])
        same_proto_mask = tf.equal(query_labels_exp, class_ids)
        diff_proto_mask = tf.logical_not(same_proto_mask)

        query_support_same_values.append(
            tf.boolean_mask(support_similarities, same_support_mask)
        )
        query_support_diff_values.append(
            tf.boolean_mask(support_similarities, diff_support_mask)
        )
        query_prototype_same_values.append(
            tf.boolean_mask(prototype_similarities, same_proto_mask)
        )
        query_prototype_diff_values.append(
            tf.boolean_mask(prototype_similarities, diff_proto_mask)
        )

    return (
        float(_mean_or_nan(tf.concat(query_support_same_values, axis=0))),
        float(_mean_or_nan(tf.concat(query_support_diff_values, axis=0))),
        float(_mean_or_nan(tf.concat(query_prototype_same_values, axis=0))),
        float(_mean_or_nan(tf.concat(query_prototype_diff_values, axis=0))),
    )


def _augment_with_gaussian_noise(
    x: tf.Tensor, noise_std: float, training: bool
) -> tf.Tensor:
    """Additive Gaussian noise augmentation used only for training updates."""
    if not training or noise_std <= 0:
        return x
    noise = tf.random.normal(shape=tf.shape(x), stddev=noise_std, dtype=x.dtype)
    return x + noise


def _compute_supervised_contrastive_loss(
    embeddings: tf.Tensor,
    labels: tf.Tensor,
    temperature: float,
) -> tf.Tensor:
    """Label-aware supervised contrastive loss over one episode."""
    normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1)
    logits = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)
    logits = logits / tf.cast(temperature, logits.dtype)

    batch_size = tf.shape(labels)[0]
    labels = tf.reshape(labels, (-1, 1))
    positive_mask = tf.cast(tf.equal(labels, tf.transpose(labels)), logits.dtype)
    logits_mask = tf.ones_like(positive_mask) - tf.eye(batch_size, dtype=logits.dtype)
    positive_mask = positive_mask * logits_mask

    logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)
    exp_logits = tf.exp(logits) * logits_mask
    log_prob = logits - tf.math.log(
        tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-8
    )

    positive_counts = tf.reduce_sum(positive_mask, axis=1)
    mean_log_prob_pos = tf.math.divide_no_nan(
        tf.reduce_sum(positive_mask * log_prob, axis=1),
        positive_counts,
    )
    valid_anchor_mask = tf.cast(positive_counts > 0, logits.dtype)

    return -tf.math.divide_no_nan(
        tf.reduce_sum(mean_log_prob_pos * valid_anchor_mask),
        tf.reduce_sum(valid_anchor_mask),
    )


def _compute_batch_all_triplet_loss(
    embeddings: tf.Tensor,
    labels: tf.Tensor,
    margin: float,
) -> tf.Tensor:
    """BatchAllTripletLoss using cosine distance d(a,b)=1-cos(a,b)."""
    normalized_embeddings = tf.nn.l2_normalize(embeddings, axis=1)
    pairwise_similarity = tf.matmul(
        normalized_embeddings, normalized_embeddings, transpose_b=True
    )
    pairwise_distance = 1.0 - pairwise_similarity

    labels = tf.reshape(labels, [-1])
    same_label = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    different_label = tf.logical_not(same_label)
    indices_not_equal = tf.logical_not(tf.eye(tf.shape(labels)[0], dtype=tf.bool))
    positive_mask = tf.logical_and(same_label, indices_not_equal)

    anchor_positive_dist = tf.expand_dims(pairwise_distance, 2)
    anchor_negative_dist = tf.expand_dims(pairwise_distance, 1)
    triplet_loss = (
        anchor_positive_dist
        - anchor_negative_dist
        + tf.cast(margin, pairwise_distance.dtype)
    )

    mask_ap = tf.expand_dims(positive_mask, 2)
    mask_an = tf.expand_dims(different_label, 1)
    triplet_mask = tf.logical_and(mask_ap, mask_an)
    triplet_loss = tf.where(
        triplet_mask,
        triplet_loss,
        tf.zeros_like(triplet_loss),
    )
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    positive_triplets = tf.cast(triplet_loss > 1e-16, pairwise_distance.dtype)
    return tf.math.divide_no_nan(
        tf.reduce_sum(triplet_loss),
        tf.reduce_sum(positive_triplets),
    )


def _build_model(
    config: PainDatasetConfig, fusion_method: str
) -> MultimodalPrototypicalNetwork:
    return MultimodalPrototypicalNetwork(
        sequence_length=config.sequence_length,
        num_sensors=len(config.sensor_idx),
        num_classes=config.n_way,
        embedding_dim=config.embedding_dim,
        modality_names=config.modality_names,
        fusion_method=fusion_method,
        distance_metric="cosine",
        classifier_mode=config.classifier_mode,
        num_tcn_blocks=config.num_tcn_blocks,
        tcn_dilation_rates=config.tcn_dilation_rates,
        tcn_kernel_size=config.tcn_kernel_size,
        strides=config.strides,
        pooling_size=config.pooling_size,
        tcn_dropout_rate=config.tcn_dropout_rate,
        tcn_attention_heads=config.tcn_attention_heads,
        tcn_attention_key_dim=config.tcn_attention_key_dim,
        tcn_attention_dropout=config.tcn_attention_dropout,
        tcn_attention_pool_size=config.tcn_attention_pool_size,
        fusion_transformer_heads=config.fusion_transformer_heads,
        fusion_transformer_layers=config.fusion_transformer_layers,
        fusion_transformer_ffn_dim=config.fusion_transformer_ffn_dim,
        fusion_ib_beta=config.fusion_ib_beta,
        filters_list=config.filters_list,
    )


def _evaluate_fixed_task_bank(
    model: MultimodalPrototypicalNetwork,
    fixed_tasks: list[dict],
    loss_fn: keras.losses.Loss,
) -> tuple[float, float, tf.Tensor, tf.Tensor]:
    losses = []
    accuracies = []
    for task in fixed_tasks:
        support_x, support_y, query_x, query_y = _to_tensors(task)
        logits = model(support_x, support_y, query_x, training=False)
        task_loss = loss_fn(query_y, logits)
        aux_loss = (
            tf.add_n(model.losses)
            if model.losses
            else tf.constant(0.0, dtype=task_loss.dtype)
        )
        loss = task_loss + aux_loss
        acc = _accuracy_from_logits(logits, query_y)
        losses.append(float(loss))
        accuracies.append(float(acc))
    bank_embeddings, bank_labels = _collect_bank_embeddings(model, fixed_tasks)
    return (
        float(np.mean(losses)),
        float(np.mean(accuracies)),
        bank_embeddings,
        bank_labels,
    )


def _sample_held_out_validation_bank(
    dataset: PainMetaDataset,
    held_out_subject: int,
    k_shot: int,
    q_query: int,
    num_val_tasks: int,
    normalize_mode: str,
    rng: np.random.Generator,
) -> list[dict]:
    return [
        dataset.sample_task(
            subject=held_out_subject,
            k_shot=k_shot,
            q_query=q_query,
            normalize_mode=normalize_mode,
            rng=rng,
        )
        for _ in range(max(1, num_val_tasks))
    ]


def _sample_same_pair_validation_bank(
    dataset: PainMetaDataset,
    support_subject: int,
    query_subject: int,
    k_shot: int,
    q_query: int,
    num_val_tasks: int,
    normalize_mode: str,
    rng: np.random.Generator,
) -> list[dict]:
    return [
        dataset.sample_task_cross_subjects(
            support_subject=support_subject,
            query_subject=query_subject,
            k_shot=k_shot,
            q_query=q_query,
            normalize_mode=normalize_mode,
            rng=rng,
        )
        for _ in range(max(1, num_val_tasks))
    ]


def _build_cross_subject_task_from_indices(
    dataset: PainMetaDataset,
    support_indices_by_class: list[np.ndarray],
    query_indices_by_class: list[np.ndarray],
    support_subject: int,
    query_subject: int,
    normalize_mode: str,
    rng: np.random.Generator,
) -> dict:
    support_X, support_y, support_subjects = [], [], []
    query_X, query_y, query_subjects = [], [], []

    for class_id in range(dataset.config.n_way):
        support_idx = np.asarray(support_indices_by_class[class_id], dtype=np.int64)
        query_idx = np.asarray(query_indices_by_class[class_id], dtype=np.int64)
        support_X.append(dataset.X[support_idx])
        support_y.append(np.full(len(support_idx), class_id, dtype=np.int32))
        support_subjects.append(
            np.full(len(support_idx), int(support_subject), dtype=np.int32)
        )
        query_X.append(dataset.X[query_idx])
        query_y.append(np.full(len(query_idx), class_id, dtype=np.int32))
        query_subjects.append(
            np.full(len(query_idx), int(query_subject), dtype=np.int32)
        )

    support_X = np.concatenate(support_X, axis=0)
    support_y = np.concatenate(support_y, axis=0)
    support_subjects = np.concatenate(support_subjects, axis=0)
    query_X = np.concatenate(query_X, axis=0)
    query_y = np.concatenate(query_y, axis=0)
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
        raise ValueError(
            f"Unknown normalize_mode: {normalize_mode}. Use 'subject', 'support', or 'none'."
        )

    support_perm = rng.permutation(len(support_y))
    query_perm = rng.permutation(len(query_y))
    return {
        "support_X": support_X[support_perm],
        "support_y": support_y[support_perm],
        "query_X": query_X[query_perm],
        "query_y": query_y[query_perm],
        "subject": -1,
        "support_subjects": support_subjects[support_perm],
        "query_subjects": query_subjects[query_perm],
    }


def _sample_disjoint_same_pair_banks(
    dataset: PainMetaDataset,
    support_subject: int,
    query_subject: int,
    k_shot: int,
    q_query: int,
    num_train_tasks: int,
    num_val_tasks: int,
    normalize_mode: str,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    total_tasks = max(1, num_train_tasks) + max(1, num_val_tasks)
    train_tasks = []
    val_tasks = []

    support_draws_by_class: list[np.ndarray] = []
    query_draws_by_class: list[np.ndarray] = []
    for class_id in range(dataset.config.n_way):
        support_indices = dataset.index[int(support_subject)][class_id]
        query_indices = dataset.index[int(query_subject)][class_id]
        required_support = total_tasks * k_shot
        required_query = total_tasks * q_query

        if len(support_indices) < required_support:
            max_total_tasks = len(support_indices) // k_shot
            raise ValueError(
                f"Cannot create disjoint train/val banks for support subject "
                f"{support_subject}, class {class_id}: need {required_support} "
                f"support samples but only {len(support_indices)} are available. "
                f"With k_shot={k_shot}, this pair supports at most {max_total_tasks} "
                f"total tasks across train and val."
            )
        if len(query_indices) < required_query:
            max_total_tasks = len(query_indices) // q_query
            raise ValueError(
                f"Cannot create disjoint train/val banks for query subject "
                f"{query_subject}, class {class_id}: need {required_query} "
                f"query samples but only {len(query_indices)} are available. "
                f"With q_query={q_query}, this pair supports at most {max_total_tasks} "
                f"total tasks across train and val."
            )

        support_draws_by_class.append(
            rng.choice(support_indices, size=required_support, replace=False)
        )
        query_draws_by_class.append(
            rng.choice(query_indices, size=required_query, replace=False)
        )

    for task_idx in range(max(1, num_train_tasks)):
        support_indices_by_class = []
        query_indices_by_class = []
        for class_id in range(dataset.config.n_way):
            support_start = task_idx * k_shot
            query_start = task_idx * q_query
            support_indices_by_class.append(
                support_draws_by_class[class_id][support_start : support_start + k_shot]
            )
            query_indices_by_class.append(
                query_draws_by_class[class_id][query_start : query_start + q_query]
            )
        train_tasks.append(
            _build_cross_subject_task_from_indices(
                dataset=dataset,
                support_indices_by_class=support_indices_by_class,
                query_indices_by_class=query_indices_by_class,
                support_subject=support_subject,
                query_subject=query_subject,
                normalize_mode=normalize_mode,
                rng=rng,
            )
        )

    train_offset = max(1, num_train_tasks)
    for val_idx in range(max(1, num_val_tasks)):
        support_indices_by_class = []
        query_indices_by_class = []
        for class_id in range(dataset.config.n_way):
            support_start = (train_offset + val_idx) * k_shot
            query_start = (train_offset + val_idx) * q_query
            support_indices_by_class.append(
                support_draws_by_class[class_id][support_start : support_start + k_shot]
            )
            query_indices_by_class.append(
                query_draws_by_class[class_id][query_start : query_start + q_query]
            )
        val_tasks.append(
            _build_cross_subject_task_from_indices(
                dataset=dataset,
                support_indices_by_class=support_indices_by_class,
                query_indices_by_class=query_indices_by_class,
                support_subject=support_subject,
                query_subject=query_subject,
                normalize_mode=normalize_mode,
                rng=rng,
            )
        )

    return train_tasks, val_tasks


def _sample_paired_cross_subject_banks(
    dataset: PainMetaDataset,
    train_subjects: list[int],
    k_shot: int,
    q_query: int,
    num_pairs: int,
    normalize_mode: str,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict], list[tuple[int, int]]]:
    if 2 * k_shot > 8:
        raise ValueError(
            f"paired same-pair train/val sampling requires 2 * k_shot <= 8, got k_shot={k_shot}"
        )
    if 2 * q_query > 8:
        raise ValueError(
            f"paired same-pair train/val sampling requires 2 * q_query <= 8, got q_query={q_query}"
        )

    train_tasks = []
    val_tasks = []
    subject_pairs = []
    for _ in range(max(1, num_pairs)):
        support_subject, query_subject = rng.choice(
            train_subjects, size=2, replace=False
        )
        support_subject = int(support_subject)
        query_subject = int(query_subject)
        train_pair_tasks, val_pair_tasks = _sample_disjoint_same_pair_banks(
            dataset=dataset,
            support_subject=support_subject,
            query_subject=query_subject,
            k_shot=k_shot,
            q_query=q_query,
            num_train_tasks=1,
            num_val_tasks=1,
            normalize_mode=normalize_mode,
            rng=rng,
        )
        train_tasks.extend(train_pair_tasks)
        val_tasks.extend(val_pair_tasks)
        subject_pairs.append((support_subject, query_subject))

    return train_tasks, val_tasks, subject_pairs


def _sample_train_task_bank(
    dataset: PainMetaDataset,
    train_subjects: list[int],
    k_shot: int,
    q_query: int,
    num_fixed_tasks: int,
    normalize_mode: str,
    subject_mode: str,
    train_subject_id: int | None,
    query_subject_id: int | None,
    rng: np.random.Generator,
) -> tuple[list[dict], list[int] | None, int | None]:
    selected_query_subject = None

    if subject_mode == "single_subject_bank":
        selected_subject = (
            int(train_subject_id)
            if train_subject_id is not None
            else int(train_subjects[0])
        )
        if selected_subject not in train_subjects:
            raise ValueError(
                f"train_subject_id={selected_subject} is not part of the train split: "
                f"{train_subjects}"
            )
        task_subjects = [selected_subject]
        fixed_tasks = [
            dataset.sample_task_from_subjects(
                subjects=task_subjects,
                k_shot=k_shot,
                q_query=q_query,
                normalize_mode=normalize_mode,
                rng=rng,
            )
            for _ in range(max(1, num_fixed_tasks))
        ]
    elif subject_mode == "cross_subject_task":
        selected_subject = (
            int(train_subject_id)
            if train_subject_id is not None
            else int(train_subjects[0])
        )
        if selected_subject not in train_subjects:
            raise ValueError(
                f"train_subject_id={selected_subject} is not part of the train split: "
                f"{train_subjects}"
            )
        remaining_subjects = [
            subject for subject in train_subjects if subject != selected_subject
        ]
        selected_query_subject = (
            int(query_subject_id)
            if query_subject_id is not None
            else int(remaining_subjects[0])
        )
        if selected_query_subject not in train_subjects:
            raise ValueError(
                f"query_subject_id={selected_query_subject} is not part of the train split: "
                f"{train_subjects}"
            )
        if selected_query_subject == selected_subject:
            raise ValueError(
                "cross_subject_task requires different support and query subjects"
            )
        task_subjects = [selected_subject, selected_query_subject]
        fixed_tasks = [
            dataset.sample_task_cross_subjects(
                support_subject=selected_subject,
                query_subject=selected_query_subject,
                k_shot=k_shot,
                q_query=q_query,
                normalize_mode=normalize_mode,
                rng=rng,
            )
            for _ in range(max(1, num_fixed_tasks))
        ]
    elif subject_mode == "cross_subject_pairs":
        task_subjects = None
        fixed_tasks = []
        for _ in range(max(1, num_fixed_tasks)):
            support_subject, query_subject = rng.choice(
                train_subjects, size=2, replace=False
            )
            fixed_tasks.append(
                dataset.sample_task_cross_subjects(
                    support_subject=int(support_subject),
                    query_subject=int(query_subject),
                    k_shot=k_shot,
                    q_query=q_query,
                    normalize_mode=normalize_mode,
                    rng=rng,
                )
            )
    else:
        task_subjects = train_subjects
        fixed_tasks = [
            dataset.sample_task_from_subjects(
                subjects=task_subjects,
                k_shot=k_shot,
                q_query=q_query,
                normalize_mode=normalize_mode,
                rng=rng,
            )
            for _ in range(max(1, num_fixed_tasks))
        ]

    return fixed_tasks, task_subjects, selected_query_subject


def run_fixed_episode_bank_overfit(
    data_dir: str,
    seed: int,
    fusion_method: str,
    num_batches: int,
    task_batch_size: int,
    learning_rate: float,
    k_shot: int,
    q_query: int,
    log_every: int,
    num_fixed_tasks: int,
    normalize_mode: str,
    classifier_mode: str,
    subject_mode: str,
    train_subject_id: int | None,
    query_subject_id: int | None,
    num_val_tasks: int,
    gaussian_noise_std: float,
    val_mode: str,
    embedding_dim: int | None = None,
    triplet_loss_weight: float | None = None,
    supcon_loss_weight: float | None = None,
    enable_window_shift_augmentation: bool | None = None,
    progress_callback: Optional[Callable[[StepMetrics], bool]] = None,
) -> tuple[list[StepMetrics], bool]:
    logger = setup_logger("fixed_episode_bank_overfit")
    num_batches = max(1, int(num_batches))
    task_batch_size = max(1, int(task_batch_size))
    config_kwargs = dict(
        seed=seed,
        deterministic_ops=True,
        k_shot=k_shot,
        q_query=q_query,
        single_loso_fold=True,
        classifier_mode=classifier_mode,
    )
    if embedding_dim is not None:
        config_kwargs["embedding_dim"] = int(embedding_dim)
    if triplet_loss_weight is not None:
        config_kwargs["triplet_loss_weight"] = float(triplet_loss_weight)
    if supcon_loss_weight is not None:
        config_kwargs["supcon_loss_weight"] = float(supcon_loss_weight)
    if enable_window_shift_augmentation is not None:
        config_kwargs["enable_window_shift_augmentation"] = bool(
            enable_window_shift_augmentation
        )
    config = PainDatasetConfig(**config_kwargs)
    set_global_reproducibility(
        seed=config.seed,
        deterministic_ops=config.deterministic_ops,
        logger=logger,
    )

    dataset = PainMetaDataset(
        data_dir=data_dir,
        config=config,
        normalize=True,
        normalize_per_subject=True,
    )
    cv = LOSOCrossValidator(dataset=dataset, seed=config.seed)
    held_out_subject = int(cv.subjects[0])
    fold = cv.get_fold(held_out_subject)
    train_subjects = [int(subject) for subject in fold["train_subjects"]]
    rng = np.random.default_rng(seed)
    fixed_tasks, task_subjects, selected_query_subject = _sample_train_task_bank(
        dataset=dataset,
        train_subjects=train_subjects,
        k_shot=k_shot,
        q_query=q_query,
        num_fixed_tasks=num_fixed_tasks,
        normalize_mode=normalize_mode,
        subject_mode=subject_mode,
        train_subject_id=train_subject_id,
        query_subject_id=query_subject_id,
        rng=rng,
    )
    first_task = fixed_tasks[0]
    logger.info(
        f"Fixed task bank sampled from train split: num_tasks={len(fixed_tasks)}, "
        f"num_batches={num_batches}, "
        f"task_batch_size={task_batch_size}, "
        f"normalize_mode={normalize_mode}, "
        f"classifier_mode={classifier_mode}, "
        f"subject_mode={subject_mode}, "
        f"gaussian_noise_std={gaussian_noise_std}, "
        f"embedding_dim={config.embedding_dim}, "
        f"triplet_loss_weight={config.triplet_loss_weight}, "
        f"supcon_loss_weight={config.supcon_loss_weight}, "
        f"enable_window_shift_augmentation={config.enable_window_shift_augmentation}, "
        f"task_subjects={task_subjects}, "
        f"query_subject={selected_query_subject}, "
        f"support_shape={first_task['support_X'].shape}, "
        f"query_shape={first_task['query_X'].shape}, "
        f"support_counts={np.bincount(first_task['support_y'], minlength=config.n_way).tolist()}, "
        f"query_counts={np.bincount(first_task['query_y'], minlength=config.n_way).tolist()}"
    )
    val_tasks = []
    if num_val_tasks > 0:
        if val_mode == "held_out_subject":
            val_tasks = _sample_held_out_validation_bank(
                dataset=dataset,
                held_out_subject=held_out_subject,
                k_shot=k_shot,
                q_query=q_query,
                num_val_tasks=num_val_tasks,
                normalize_mode=normalize_mode,
                rng=rng,
            )
            logger.info(
                f"Held-out validation bank sampled: num_tasks={len(val_tasks)}, "
                f"held_out_subject={held_out_subject}, normalize_mode={normalize_mode}"
            )
        elif val_mode == "same_subject_pair":
            if subject_mode == "cross_subject_task":
                if task_subjects is None or selected_query_subject is None:
                    raise ValueError(
                        "same_subject_pair validation requires explicit support/query train subjects"
                    )
                fixed_tasks, val_tasks = _sample_disjoint_same_pair_banks(
                    dataset=dataset,
                    support_subject=int(task_subjects[0]),
                    query_subject=int(selected_query_subject),
                    k_shot=k_shot,
                    q_query=q_query,
                    num_train_tasks=num_fixed_tasks,
                    num_val_tasks=num_val_tasks,
                    normalize_mode=normalize_mode,
                    rng=rng,
                )
                logger.info(
                    f"Disjoint same-pair train/validation banks sampled: "
                    f"train_tasks={len(fixed_tasks)}, val_tasks={len(val_tasks)}, "
                    f"support_subject={int(task_subjects[0])}, "
                    f"query_subject={int(selected_query_subject)}, "
                    f"normalize_mode={normalize_mode}"
                )
            elif subject_mode == "cross_subject_pairs":
                if num_val_tasks != num_fixed_tasks:
                    raise ValueError(
                        "For subject_mode='cross_subject_pairs' and val_mode='same_subject_pair', "
                        "num_val_tasks must equal num_fixed_tasks so each train pair gets one disjoint val task."
                    )
                fixed_tasks, val_tasks, subject_pairs = (
                    _sample_paired_cross_subject_banks(
                        dataset=dataset,
                        train_subjects=train_subjects,
                        k_shot=k_shot,
                        q_query=q_query,
                        num_pairs=num_fixed_tasks,
                        normalize_mode=normalize_mode,
                        rng=rng,
                    )
                )
                logger.info(
                    f"Paired cross-subject train/validation banks sampled: "
                    f"num_pairs={len(subject_pairs)}, normalize_mode={normalize_mode}, "
                    f"subject_pairs={subject_pairs}"
                )
            else:
                raise ValueError(
                    "val_mode='same_subject_pair' requires subject_mode='cross_subject_task' "
                    "or subject_mode='cross_subject_pairs'"
                )
        else:
            raise ValueError(
                f"Unknown val_mode: {val_mode}. Use 'held_out_subject' or 'same_subject_pair'."
            )

    model = _build_model(config=config, fusion_method=fusion_method)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    history: list[StepMetrics] = []
    for batch_step in range(1, num_batches + 1):
        batch_start = (batch_step - 1) * task_batch_size
        batch_tasks = [
            fixed_tasks[(batch_start + i) % len(fixed_tasks)]
            for i in range(task_batch_size)
        ]

        with tf.GradientTape() as tape:
            per_task_losses = []
            for task in batch_tasks:
                support_x, support_y, query_x, query_y = _to_tensors(task)
                support_x_aug = _augment_with_gaussian_noise(
                    support_x, noise_std=gaussian_noise_std, training=True
                )
                query_x_aug = _augment_with_gaussian_noise(
                    query_x, noise_std=gaussian_noise_std, training=True
                )
                episode_outputs = model.forward_episode(
                    support_x=support_x_aug,
                    support_y=support_y,
                    query_x=query_x_aug,
                    training=True,
                )
                logits = episode_outputs["logits"]
                task_loss = loss_fn(query_y, logits)
                aux_loss = (
                    tf.add_n(model.losses)
                    if model.losses
                    else tf.constant(0.0, dtype=task_loss.dtype)
                )
                support_embeddings = episode_outputs["support_embeddings"]
                query_embeddings = episode_outputs["query_embeddings"]
                all_embeddings = tf.concat(
                    [support_embeddings, query_embeddings], axis=0
                )
                all_labels = tf.concat([support_y, query_y], axis=0)

                if config.supcon_loss_weight > 0:
                    supcon_loss = _compute_supervised_contrastive_loss(
                        embeddings=all_embeddings,
                        labels=all_labels,
                        temperature=float(config.supcon_temperature),
                    )
                else:
                    supcon_loss = tf.constant(0.0, dtype=task_loss.dtype)

                if config.triplet_loss_weight > 0:
                    triplet_loss = _compute_batch_all_triplet_loss(
                        embeddings=all_embeddings,
                        labels=all_labels,
                        margin=float(config.triplet_margin),
                    )
                else:
                    triplet_loss = tf.constant(0.0, dtype=task_loss.dtype)

                total_loss = (
                    task_loss
                    + aux_loss
                    + tf.cast(config.supcon_loss_weight, task_loss.dtype) * supcon_loss
                    + tf.cast(config.triplet_loss_weight, task_loss.dtype)
                    * triplet_loss
                )
                per_task_losses.append(total_loss)

            batch_loss = tf.reduce_mean(tf.stack(per_task_losses))

        grads = tape.gradient(batch_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        report_due = (
            batch_step == 1 or batch_step % log_every == 0 or batch_step == num_batches
        )
        if not report_due:
            continue

        (
            eval_loss,
            eval_acc,
            train_bank_embeddings,
            train_bank_labels,
        ) = _evaluate_fixed_task_bank(
            model=model,
            fixed_tasks=fixed_tasks,
            loss_fn=loss_fn,
        )
        train_within_class_cosine, train_between_class_cosine = (
            _pairwise_bank_cosine_stats(
                embeddings=train_bank_embeddings,
                labels=train_bank_labels,
            )
        )
        (
            train_query_support_same_class_cosine,
            train_query_support_diff_class_cosine,
            train_query_prototype_same_class_cosine,
            train_query_prototype_diff_class_cosine,
        ) = _task_local_cosine_stats(model=model, fixed_tasks=fixed_tasks)
        val_loss = None
        val_acc = None
        val_within_class_cosine = None
        val_between_class_cosine = None
        cross_bank_same_class_cosine = None
        cross_bank_diff_class_cosine = None
        val_query_support_same_class_cosine = None
        val_query_support_diff_class_cosine = None
        val_query_prototype_same_class_cosine = None
        val_query_prototype_diff_class_cosine = None
        if val_tasks:
            (
                val_loss,
                val_acc,
                val_bank_embeddings,
                val_bank_labels,
            ) = _evaluate_fixed_task_bank(
                model=model,
                fixed_tasks=val_tasks,
                loss_fn=loss_fn,
            )
            val_within_class_cosine, val_between_class_cosine = (
                _pairwise_bank_cosine_stats(
                    embeddings=val_bank_embeddings,
                    labels=val_bank_labels,
                )
            )
            (
                val_query_support_same_class_cosine,
                val_query_support_diff_class_cosine,
                val_query_prototype_same_class_cosine,
                val_query_prototype_diff_class_cosine,
            ) = _task_local_cosine_stats(model=model, fixed_tasks=val_tasks)
            cross_bank_same_class_cosine, cross_bank_diff_class_cosine = (
                _cross_bank_cosine_stats(
                    train_embeddings=train_bank_embeddings,
                    train_labels=train_bank_labels,
                    val_embeddings=val_bank_embeddings,
                    val_labels=val_bank_labels,
                )
            )

        metrics = StepMetrics(
            step=batch_step,
            loss=eval_loss,
            accuracy=eval_acc,
            heldout_val_loss=val_loss,
            heldout_val_accuracy=val_acc,
            within_class_cosine=train_within_class_cosine,
            between_class_cosine=train_between_class_cosine,
            cross_bank_same_class_cosine=cross_bank_same_class_cosine,
            cross_bank_diff_class_cosine=cross_bank_diff_class_cosine,
            task_query_support_same_class_cosine=train_query_support_same_class_cosine,
            task_query_support_diff_class_cosine=train_query_support_diff_class_cosine,
            task_query_prototype_same_class_cosine=train_query_prototype_same_class_cosine,
            task_query_prototype_diff_class_cosine=train_query_prototype_diff_class_cosine,
        )
        history.append(metrics)
        if progress_callback is not None and bool(progress_callback(metrics)):
            logger.info(
                f"Early stop requested by progress_callback at batch {metrics.step}."
            )
            break

        message = (
            f"Batch {batch_step}/{num_batches} | train_bank_loss={metrics.loss:.4f}, "
            f"train_bank_accuracy={metrics.accuracy:.4f}, "
            f"train_within_class_cosine={train_within_class_cosine:.4f}, "
            f"train_between_class_cosine={train_between_class_cosine:.4f}, "
            f"train_query_support_same_class_cosine={train_query_support_same_class_cosine:.4f}, "
            f"train_query_support_diff_class_cosine={train_query_support_diff_class_cosine:.4f}, "
            f"train_query_prototype_same_class_cosine={train_query_prototype_same_class_cosine:.4f}, "
            f"train_query_prototype_diff_class_cosine={train_query_prototype_diff_class_cosine:.4f}"
        )
        if val_loss is not None and val_acc is not None:
            message += (
                f", heldout_val_loss={val_loss:.4f}, "
                f"heldout_val_accuracy={val_acc:.4f}, "
                f"heldout_within_class_cosine={val_within_class_cosine:.4f}, "
                f"heldout_between_class_cosine={val_between_class_cosine:.4f}, "
                f"heldout_query_support_same_class_cosine={val_query_support_same_class_cosine:.4f}, "
                f"heldout_query_support_diff_class_cosine={val_query_support_diff_class_cosine:.4f}, "
                f"heldout_query_prototype_same_class_cosine={val_query_prototype_same_class_cosine:.4f}, "
                f"heldout_query_prototype_diff_class_cosine={val_query_prototype_diff_class_cosine:.4f}, "
                f"cross_bank_same_class_cosine={cross_bank_same_class_cosine:.4f}, "
                f"cross_bank_diff_class_cosine={cross_bank_diff_class_cosine:.4f}"
            )
        logger.info(message)

    first = history[0]
    last = history[-1]
    memorized = last.accuracy >= 0.95 and last.loss < first.loss * 0.5
    logger.info(
        "Fixed-bank overfit check: "
        f"first_loss={first.loss:.4f}, last_loss={last.loss:.4f}, "
        f"first_acc={first.accuracy:.4f}, last_acc={last.accuracy:.4f}, "
        f"passed={memorized}"
    )
    return history, memorized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overfit a fixed bank of prototypical episodes with CE-only training."
    )
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fusion-method",
        type=str,
        default="gated",
        choices=("mean", "gated", "transformer_ib"),
        help="Embedding fusion mode to use for the fixed-bank overfit experiment.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Number of optimization batches (gradient updates).",
    )
    parser.add_argument(
        "--task-batch-size",
        type=int,
        default=12,
        help="Number of episodic tasks per optimization batch.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.0006)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--q-query", type=int, default=5)
    parser.add_argument("--num-fixed-tasks", type=int, default=96)
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="subject",
        choices=("subject", "support", "none"),
    )
    parser.add_argument(
        "--classifier-mode",
        type=str,
        default="prototype",
        choices=("prototype", "soft_knn"),
    )
    parser.add_argument(
        "--subject-mode",
        type=str,
        default="mixed",
        choices=(
            "mixed",
            "single_subject_bank",
            "cross_subject_task",
            "cross_subject_pairs",
        ),
        help="Sample fixed-bank tasks from all train subjects, one train subject only, a single fixed support/query pair, or many support/query pairs.",
    )
    parser.add_argument(
        "--train-subject-id",
        type=int,
        default=1,
        help="Train subject to use when --subject-mode=single_subject_bank. Defaults to the first subject in the train split.",
    )
    parser.add_argument(
        "--query-subject-id",
        type=int,
        default=2,
        help="Query subject to use when --subject-mode=cross_subject_task. Defaults to the first train subject different from --train-subject-id.",
    )
    parser.add_argument(
        "--num-val-tasks",
        type=int,
        default=30,
        help="Number of fixed held-out-subject validation tasks to evaluate alongside the train bank. Use 0 to disable.",
    )
    parser.add_argument(
        "--val-mode",
        type=str,
        default="held_out_subject",
        choices=("held_out_subject", "same_subject_pair"),
        help="Validation bank source: LOSO held-out subject or freshly resampled tasks from the same training support/query subject pair.",
    )
    parser.add_argument(
        "--gaussian-noise-std",
        type=float,
        default=0.02,
        help="Stddev of additive Gaussian noise used only during training updates.",
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if the model fails to memorize the fixed task bank.",
    )
    args = parser.parse_args()

    _, memorized = run_fixed_episode_bank_overfit(
        data_dir=args.data_dir,
        seed=args.seed,
        fusion_method=args.fusion_method,
        num_batches=args.num_batches,
        task_batch_size=args.task_batch_size,
        learning_rate=args.learning_rate,
        k_shot=args.k_shot,
        q_query=args.q_query,
        log_every=args.log_every,
        num_fixed_tasks=args.num_fixed_tasks,
        normalize_mode=args.normalize_mode,
        classifier_mode=args.classifier_mode,
        subject_mode=args.subject_mode,
        train_subject_id=args.train_subject_id,
        query_subject_id=args.query_subject_id,
        num_val_tasks=args.num_val_tasks,
        gaussian_noise_std=args.gaussian_noise_std,
        val_mode=args.val_mode,
    )
    if args.strict and not memorized:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
