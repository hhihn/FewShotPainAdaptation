import argparse
import json
from pathlib import Path
import sys

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
    )


def _task_subject_summary(task: dict, n_way: int) -> dict:
    summary = {
        "support_subject_counts": {
            int(subject): int(count)
            for subject, count in zip(
                *np.unique(task["support_subjects"], return_counts=True)
            )
        },
        "query_subject_counts": {
            int(subject): int(count)
            for subject, count in zip(
                *np.unique(task["query_subjects"], return_counts=True)
            )
        },
        "support_class_subject_counts": {},
        "query_class_subject_counts": {},
    }
    for class_id in range(n_way):
        support_mask = task["support_y"] == class_id
        query_mask = task["query_y"] == class_id
        summary["support_class_subject_counts"][class_id] = {
            int(subject): int(count)
            for subject, count in zip(
                *np.unique(task["support_subjects"][support_mask], return_counts=True)
            )
        }
        summary["query_class_subject_counts"][class_id] = {
            int(subject): int(count)
            for subject, count in zip(
                *np.unique(task["query_subjects"][query_mask], return_counts=True)
            )
        }
    return summary


def _train_on_fixed_bank(
    model: MultimodalPrototypicalNetwork,
    fixed_tasks: list[dict],
    steps: int,
    learning_rate: float,
    log_every: int,
    logger,
) -> None:
    if steps <= 0:
        return

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for step in range(1, steps + 1):
        task = fixed_tasks[(step - 1) % len(fixed_tasks)]
        support_x = tf.constant(task["support_X"], dtype=tf.float32)
        support_y = tf.constant(task["support_y"], dtype=tf.int32)
        query_x = tf.constant(task["query_X"], dtype=tf.float32)
        query_y = tf.constant(task["query_y"], dtype=tf.int32)

        with tf.GradientTape() as tape:
            logits = model(support_x, support_y, query_x, training=True)
            task_loss = loss_fn(query_y, logits)
            aux_loss = (
                tf.add_n(model.losses)
                if model.losses
                else tf.constant(0.0, dtype=task_loss.dtype)
            )
            loss = task_loss + aux_loss

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step == 1 or step % log_every == 0 or step == steps:
            preds = tf.argmax(logits, axis=1, output_type=tf.int32)
            acc = tf.reduce_mean(tf.cast(tf.equal(preds, query_y), tf.float32))
            logger.info(
                f"Train step {step}/{steps} | task_loss={float(task_loss):.4f}, accuracy={float(acc):.4f}"
            )


def inspect_fixed_task_bank(
    data_dir: str,
    output_path: str,
    seed: int,
    fusion_method: str,
    num_fixed_tasks: int,
    k_shot: int,
    q_query: int,
    train_steps: int,
    learning_rate: float,
    log_every: int,
    normalize_mode: str,
) -> str:
    logger = setup_logger("inspect_fixed_task_bank")
    config = PainDatasetConfig(
        seed=seed,
        deterministic_ops=True,
        k_shot=k_shot,
        q_query=q_query,
        single_loso_fold=True,
        supcon_loss_weight=0.0,
    )
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

    fixed_tasks = [
        dataset.sample_task_from_subjects(
            subjects=train_subjects,
            k_shot=k_shot,
            q_query=q_query,
            normalize_mode=normalize_mode,
            rng=rng,
            include_sample_subjects=True,
        )
        for _ in range(max(1, num_fixed_tasks))
    ]

    for task_idx, task in enumerate(fixed_tasks):
        logger.info(
            f"Task {task_idx}: "
            f"{json.dumps(_task_subject_summary(task, config.n_way), sort_keys=True)}"
        )

    model = _build_model(config=config, fusion_method=fusion_method)
    _train_on_fixed_bank(
        model=model,
        fixed_tasks=fixed_tasks,
        steps=train_steps,
        learning_rate=learning_rate,
        log_every=log_every,
        logger=logger,
    )

    all_embeddings = []
    all_labels = []
    all_subjects = []
    all_task_ids = []
    all_splits = []
    all_prototypes = []
    all_prototype_task_ids = []
    all_prototype_labels = []

    for task_idx, task in enumerate(fixed_tasks):
        support_x = tf.constant(task["support_X"], dtype=tf.float32)
        support_y = tf.constant(task["support_y"], dtype=tf.int32)
        query_x = tf.constant(task["query_X"], dtype=tf.float32)

        episode_outputs = model.forward_episode(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            training=False,
        )
        support_embeddings = episode_outputs["support_embeddings"].numpy()
        query_embeddings = episode_outputs["query_embeddings"].numpy()
        prototypes = episode_outputs["prototypes"].numpy()

        all_embeddings.append(support_embeddings)
        all_embeddings.append(query_embeddings)
        all_labels.append(task["support_y"])
        all_labels.append(task["query_y"])
        all_subjects.append(task["support_subjects"])
        all_subjects.append(task["query_subjects"])
        all_task_ids.append(np.full(len(task["support_y"]), task_idx, dtype=np.int32))
        all_task_ids.append(np.full(len(task["query_y"]), task_idx, dtype=np.int32))
        all_splits.append(np.array(["support"] * len(task["support_y"]), dtype="<U8"))
        all_splits.append(np.array(["query"] * len(task["query_y"]), dtype="<U8"))
        all_prototypes.append(prototypes)
        all_prototype_task_ids.append(
            np.full(config.n_way, task_idx, dtype=np.int32)
        )
        all_prototype_labels.append(np.arange(config.n_way, dtype=np.int32))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        embeddings=np.concatenate(all_embeddings, axis=0),
        labels=np.concatenate(all_labels, axis=0),
        subjects=np.concatenate(all_subjects, axis=0),
        task_ids=np.concatenate(all_task_ids, axis=0),
        splits=np.concatenate(all_splits, axis=0),
        prototypes=np.concatenate(all_prototypes, axis=0),
        prototype_task_ids=np.concatenate(all_prototype_task_ids, axis=0),
        prototype_labels=np.concatenate(all_prototype_labels, axis=0),
    )
    logger.info(f"Saved embedding inspection bundle to {output}")
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a fixed task bank for subject composition and embedding structure."
    )
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/diagnostics/fixed_task_bank_embeddings.npz",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fusion-method", type=str, default="mean")
    parser.add_argument("--num-fixed-tasks", type=int, default=1)
    parser.add_argument("--k-shot", type=int, default=3)
    parser.add_argument("--q-query", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=125)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="none",
        choices=("subject", "support", "none"),
    )
    args = parser.parse_args()

    inspect_fixed_task_bank(
        data_dir=args.data_dir,
        output_path=args.output_path,
        seed=args.seed,
        fusion_method=args.fusion_method,
        num_fixed_tasks=args.num_fixed_tasks,
        k_shot=args.k_shot,
        q_query=args.q_query,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        normalize_mode=args.normalize_mode,
    )


if __name__ == "__main__":
    main()
