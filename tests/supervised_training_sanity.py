import argparse
from dataclasses import dataclass
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


@dataclass
class EpochMetrics:
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


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


def _build_model(config: PainDatasetConfig, fusion_method: str) -> MultimodalPrototypicalNetwork:
    return MultimodalPrototypicalNetwork(
        sequence_length=config.sequence_length,
        num_sensors=len(config.sensor_idx),
        num_classes=config.num_stimuli_levels,
        embedding_dim=config.embedding_dim,
        modality_names=config.modality_names,
        fusion_method=fusion_method,
        distance_metric="cosine",
        num_tcn_blocks=config.num_tcn_blocks,
        tcn_attention_pool_size=config.tcn_attention_pool_size,
        fusion_transformer_heads=config.fusion_transformer_heads,
        fusion_transformer_layers=config.fusion_transformer_layers,
        fusion_transformer_ffn_dim=config.fusion_transformer_ffn_dim,
        fusion_ib_beta=config.fusion_ib_beta,
    )


def run_sanity_training(
    data_dir: str,
    seed: int,
    fusion_method: str,
    num_epochs: int,
    tasks_per_epoch: int,
    val_tasks: int,
    k_shot: int,
    q_query: int,
    learning_rate: float,
) -> tuple[list[EpochMetrics], bool]:
    logger = setup_logger("supervised_training_sanity")
    config = PainDatasetConfig(
        seed=seed,
        deterministic_ops=True,
        k_shot=k_shot,
        q_query=q_query,
        num_epochs=num_epochs,
        tasks_per_epoch=tasks_per_epoch,
        val_tasks=val_tasks,
        single_loso_fold=True,
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
    train_sampler = fold["train_sampler"]
    val_sampler = fold["val_sampler"]

    model = _build_model(config=config, fusion_method=fusion_method)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    history: list[EpochMetrics] = []
    for epoch in range(1, config.num_epochs + 1):
        train_losses = []
        train_accs = []
        for _ in range(config.tasks_per_epoch):
            task = train_sampler.get_task()
            support_x, support_y, query_x, query_y = _to_tensors(task)
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
            train_losses.append(float(loss))
            train_accs.append(float(_accuracy_from_logits(logits, query_y)))

        val_losses = []
        val_accs = []
        for _ in range(config.val_tasks):
            task = val_sampler.get_task()
            support_x, support_y, query_x, query_y = _to_tensors(task)
            logits = model(support_x, support_y, query_x, training=False)
            task_loss = loss_fn(query_y, logits)
            aux_loss = (
                tf.add_n(model.losses)
                if model.losses
                else tf.constant(0.0, dtype=task_loss.dtype)
            )
            loss = task_loss + aux_loss
            val_losses.append(float(loss))
            val_accs.append(float(_accuracy_from_logits(logits, query_y)))

        metrics = EpochMetrics(
            train_loss=float(np.mean(train_losses)),
            train_acc=float(np.mean(train_accs)),
            val_loss=float(np.mean(val_losses)),
            val_acc=float(np.mean(val_accs)),
        )
        history.append(metrics)
        logger.info(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"train_loss={metrics.train_loss:.4f}, train_acc={metrics.train_acc:.4f} | "
            f"val_loss={metrics.val_loss:.4f}, val_acc={metrics.val_acc:.4f}"
        )

    first = history[0]
    last = history[-1]
    learned = (last.train_loss < first.train_loss * 0.98) or (
        last.train_acc > first.train_acc + 0.05
    )
    logger.info(
        "Learning signal check: "
        f"first_train_loss={first.train_loss:.4f}, last_train_loss={last.train_loss:.4f}, "
        f"first_train_acc={first.train_acc:.4f}, last_train_acc={last.train_acc:.4f}, "
        f"passed={learned}"
    )
    return history, learned


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check learning with native prototypical classification."
    )
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fusion-method", type=str, default="mean")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--tasks-per-epoch", type=int, default=100)
    parser.add_argument("--val-tasks", type=int, default=20)
    parser.add_argument("--k-shot", type=int, default=3)
    parser.add_argument("--q-query", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if no learning signal is detected.",
    )
    args = parser.parse_args()

    _, learned = run_sanity_training(
        data_dir=args.data_dir,
        seed=args.seed,
        fusion_method=args.fusion_method,
        num_epochs=args.epochs,
        tasks_per_epoch=args.tasks_per_epoch,
        val_tasks=args.val_tasks,
        k_shot=args.k_shot,
        q_query=args.q_query,
        learning_rate=args.learning_rate,
    )
    if args.strict and not learned:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
