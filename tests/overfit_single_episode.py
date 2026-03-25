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
class StepMetrics:
    step: int
    loss: float
    accuracy: float


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
    )


def _evaluate_fixed_task_bank(
    model: MultimodalPrototypicalNetwork,
    fixed_tasks: list[dict],
    loss_fn: keras.losses.Loss,
) -> tuple[float, float]:
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
    return float(np.mean(losses)), float(np.mean(accuracies))


def run_fixed_episode_bank_overfit(
    data_dir: str,
    seed: int,
    fusion_method: str,
    steps: int,
    learning_rate: float,
    k_shot: int,
    q_query: int,
    log_every: int,
    num_fixed_tasks: int,
    normalize_mode: str,
    classifier_mode: str,
) -> tuple[list[StepMetrics], bool]:
    logger = setup_logger("fixed_episode_bank_overfit")
    config = PainDatasetConfig(
        seed=seed,
        deterministic_ops=True,
        k_shot=k_shot,
        q_query=q_query,
        single_loso_fold=True,
        classifier_mode=classifier_mode,
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
        )
        for _ in range(max(1, num_fixed_tasks))
    ]
    first_task = fixed_tasks[0]
    logger.info(
        f"Fixed task bank sampled from train split: num_tasks={len(fixed_tasks)}, "
        f"normalize_mode={normalize_mode}, "
        f"classifier_mode={classifier_mode}, "
        f"support_shape={first_task['support_X'].shape}, "
        f"query_shape={first_task['query_X'].shape}, "
        f"support_counts={np.bincount(first_task['support_y'], minlength=config.n_way).tolist()}, "
        f"query_counts={np.bincount(first_task['query_y'], minlength=config.n_way).tolist()}"
    )

    model = _build_model(config=config, fusion_method=fusion_method)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    history: list[StepMetrics] = []
    for step in range(1, steps + 1):
        task = fixed_tasks[(step - 1) % len(fixed_tasks)]
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

        eval_loss, eval_acc = _evaluate_fixed_task_bank(
            model=model,
            fixed_tasks=fixed_tasks,
            loss_fn=loss_fn,
        )

        metrics = StepMetrics(
            step=step,
            loss=eval_loss,
            accuracy=eval_acc,
        )
        history.append(metrics)

        if step == 1 or step % log_every == 0 or step == steps:
            logger.info(
                f"Step {step}/{steps} | bank_loss={metrics.loss:.4f}, "
                f"bank_accuracy={metrics.accuracy:.4f}"
            )

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
    parser.add_argument("--fusion-method", type=str, default="mean")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--k-shot", type=int, default=3)
    parser.add_argument("--q-query", type=int, default=3)
    parser.add_argument("--num-fixed-tasks", type=int, default=10)
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
    parser.add_argument("--log-every", type=int, default=25)
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
        steps=args.steps,
        learning_rate=args.learning_rate,
        k_shot=args.k_shot,
        q_query=args.q_query,
        log_every=args.log_every,
        num_fixed_tasks=args.num_fixed_tasks,
        normalize_mode=args.normalize_mode,
        classifier_mode=args.classifier_mode,
    )
    if args.strict and not memorized:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
