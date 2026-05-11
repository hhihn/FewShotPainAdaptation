import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_loaders.pain_ds_config import PainDatasetConfig
from learner.few_shot_pain_learner import FewShotPainLearner


def _write_tiny_dataset(data_dir: Path) -> None:
    rng = np.random.default_rng(1001)
    x_rows = []
    y_rows = []
    subject_rows = []
    for subject in range(3):
        for class_id in range(2):
            for _ in range(3):
                x_rows.append(
                    rng.normal(size=(32, 6, 1)).astype(np.float32)
                    + float(class_id)
                )
                label = np.zeros(2, dtype=np.float32)
                label[class_id] = 1.0
                y_rows.append(label)
                subject_rows.append(subject)

    np.save(data_dir / "X_pre.npy", np.stack(x_rows, axis=0))
    np.save(data_dir / "y_heater.npy", np.stack(y_rows, axis=0))
    np.save(data_dir / "subjects.npy", np.array(subject_rows, dtype=np.int32))


def _make_tiny_learner(data_dir: Path) -> FewShotPainLearner:
    config = PainDatasetConfig(
        dataset_source="painmonit",
        sequence_length=32,
        num_stimuli_levels=2,
        task_class_ids=(0, 1),
        k_shot=1,
        q_query=1,
        train_batch_size=1,
        embedding_batch_size=1,
        num_epochs=1,
        tasks_per_epoch=1,
        val_tasks=1,
        heldout_eval_tasks=1,
        k_shot_adaptation_steps=0,
        embedding_dim=4,
        eegnet_temporal_filters=2,
        eegnet_depth_multiplier=1,
        eegnet_separable_filters=4,
        eegnet_temporal_kernel_size=8,
        eegnet_separable_kernel_size=4,
        eegnet_pool_size_1=2,
        eegnet_pool_size_2=2,
        eegnet_dropout_rate=0.0,
        triplet_loss_weight=0.0,
        gaussian_noise_std=0.0,
        enable_window_shift_augmentation=False,
        deterministic_ops=False,
        seed=11,
    )
    return FewShotPainLearner(
        config=config,
        data_dir=str(data_dir),
        learning_rate=1e-3,
    )


def _episode_batch(learner: FewShotPainLearner):
    rng = np.random.default_rng(2026)
    class_ids = np.arange(learner.config.n_way, dtype=np.int32)
    support_y = np.repeat(class_ids, learner.config.k_shot)[np.newaxis, :]
    query_y = np.repeat(class_ids, learner.config.q_query)[np.newaxis, :]
    support_x = rng.normal(
        size=(
            1,
            learner.support_size,
            learner.sequence_length,
            learner.num_sensors,
        )
    ).astype(np.float32)
    query_x = rng.normal(
        size=(
            1,
            learner.query_size,
            learner.sequence_length,
            learner.num_sensors,
        )
    ).astype(np.float32)
    return (
        tf.constant(support_x, dtype=tf.float32),
        tf.constant(support_y, dtype=tf.int32),
        tf.constant(query_x, dtype=tf.float32),
        tf.constant(query_y, dtype=tf.int32),
    )


class EpisodicLearningEngineResetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)
        _write_tiny_dataset(self.data_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reset_restores_initial_state_without_recreating_compiled_functions(self):
        learner = _make_tiny_learner(self.data_dir)
        engine = learner.engine
        train_fn = engine._compiled_train_batch_step
        eval_fn = engine._compiled_eval_batch_step
        initial_model_weights = [tf.identity(value) for value in engine.model.weights]
        initial_optimizer_variables = [
            tf.identity(value) for value in engine.optimizer.variables
        ]

        engine.train_batch_step_tensors(*_episode_batch(learner))
        self.assertGreater(int(engine.optimizer.iterations.numpy()), 0)

        engine.reset_model_state_for_new_fold()

        self.assertIs(engine._compiled_train_batch_step, train_fn)
        self.assertIs(engine._compiled_eval_batch_step, eval_fn)
        self.assertIs(learner._compiled_train_batch_step, train_fn)
        self.assertIs(learner._compiled_eval_batch_step, eval_fn)
        for variable, expected in zip(engine.model.weights, initial_model_weights):
            np.testing.assert_allclose(variable.numpy(), expected.numpy(), atol=1e-7)
        for variable, expected in zip(
            engine.optimizer.variables,
            initial_optimizer_variables,
        ):
            np.testing.assert_allclose(variable.numpy(), expected.numpy(), atol=1e-7)

    def test_reset_cycles_do_not_increase_train_function_tracing_count(self):
        learner = _make_tiny_learner(self.data_dir)
        engine = learner.engine
        batch = _episode_batch(learner)

        engine.train_batch_step_tensors(*batch)
        tracing_count = (
            engine._compiled_train_batch_step.experimental_get_tracing_count()
        )

        for _ in range(3):
            engine.reset_model_state_for_new_fold()
            engine.train_batch_step_tensors(*batch)

        self.assertEqual(
            engine._compiled_train_batch_step.experimental_get_tracing_count(),
            tracing_count,
        )


if __name__ == "__main__":
    unittest.main()
