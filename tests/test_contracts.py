import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_loaders.loso_cross_validator import LOSOCrossValidator
from data_loaders.pain_ds_config import PainDatasetConfig
from data_loaders.pain_meta_dataset import PainMetaDataset
from learner.few_shot_pain_learner import FewShotPainLearner


def _write_synthetic_dataset(
    data_dir: Path,
    num_subjects: int = 3,
    n_way: int = 6,
    reps_per_class: int = 4,
    seq_len: int = 2500,
    full_sensor_count: int = 6,
) -> None:
    """Create a small deterministic dataset compatible with sensor_idx=[1,4,5]."""
    rng = np.random.default_rng(123)

    X_rows = []
    y_rows = []
    subject_rows = []

    for subject in range(num_subjects):
        for class_id in range(n_way):
            for _ in range(reps_per_class):
                x = rng.normal(size=(seq_len, full_sensor_count, 1)).astype(np.float32)
                y = np.zeros(n_way, dtype=np.float32)
                y[class_id] = 1.0

                X_rows.append(x)
                y_rows.append(y)
                subject_rows.append(subject)

    X = np.stack(X_rows, axis=0)
    y = np.stack(y_rows, axis=0)
    subjects = np.array(subject_rows, dtype=np.int32)

    np.save(data_dir / "X_pre.npy", X)
    np.save(data_dir / "y_heater.npy", y)
    np.save(data_dir / "subjects.npy", subjects)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)
        _write_synthetic_dataset(self.data_dir)

        self.config = PainDatasetConfig(
            sequence_length=2500,
            n_way=6,
            num_stimuli_levels=6,
            k_shot=1,
            q_query=1,
            seed=7,
            deterministic_ops=True,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_loso_split_contracts(self):
        dataset = PainMetaDataset(data_dir=str(self.data_dir), config=self.config)
        cv = LOSOCrossValidator(
            dataset=dataset,
            k_shot=self.config.k_shot,
            q_query=self.config.q_query,
            tasks_per_epoch=2,
            seed=self.config.seed,
        )

        all_subjects = set(dataset.unique_subjects.tolist())

        for test_subject in dataset.unique_subjects:
            fold = cv.get_fold(int(test_subject))
            train_subjects = set(fold["train_subjects"])
            val_subjects = set(fold["val_subjects"])

            self.assertNotIn(test_subject, train_subjects)
            self.assertNotIn(test_subject, val_subjects)
            self.assertTrue(train_subjects.isdisjoint(val_subjects))
            self.assertEqual(
                train_subjects | val_subjects, all_subjects - {int(test_subject)}
            )
            self.assertEqual(fold["n_train_subjects"], len(train_subjects))
            self.assertEqual(fold["n_val_subjects"], len(val_subjects))

    def test_sampler_task_contracts(self):
        dataset = PainMetaDataset(data_dir=str(self.data_dir), config=self.config)
        cv = LOSOCrossValidator(
            dataset=dataset,
            k_shot=self.config.k_shot,
            q_query=self.config.q_query,
            tasks_per_epoch=2,
            seed=self.config.seed,
        )

        fold = cv.get_fold(test_subject=int(dataset.unique_subjects[0]))
        samplers = (
            fold["train_sampler"],
            fold["val_sampler"],
            fold["test_sampler"],
        )

        for sampler in samplers:
            task = sampler.get_task()
            self.assertEqual(
                task["support_X"].shape,
                (
                    self.config.n_way * self.config.k_shot,
                    self.config.sequence_length,
                    len(self.config.sensor_idx),
                ),
            )
            self.assertEqual(
                task["query_X"].shape,
                (
                    self.config.n_way * self.config.q_query,
                    self.config.sequence_length,
                    len(self.config.sensor_idx),
                ),
            )
            self.assertEqual(
                task["support_y"].shape, (self.config.n_way * self.config.k_shot,)
            )
            self.assertEqual(
                task["query_y"].shape, (self.config.n_way * self.config.q_query,)
            )
            self.assertIn("subject", task)

            support_counts = np.bincount(
                task["support_y"], minlength=self.config.n_way
            )
            query_counts = np.bincount(task["query_y"], minlength=self.config.n_way)
            self.assertTrue(np.all(support_counts == self.config.k_shot))
            self.assertTrue(np.all(query_counts == self.config.q_query))

    def test_training_contracts_smoke(self):
        learner = FewShotPainLearner(
            config=self.config,
            data_dir=str(self.data_dir),
            learning_rate=1e-3,
            fusion_method="mean",
            seed=self.config.seed,
            deterministic_ops=self.config.deterministic_ops,
        )

        results = learner.train(num_epochs=1, tasks_per_epoch=1, val_tasks=1)

        required_keys = {
            "train_losses",
            "train_accuracies",
            "val_losses",
            "val_accuracies",
            "test_losses",
            "test_accuracies",
        }
        self.assertTrue(required_keys.issubset(results.keys()))

        n_folds = len(learner.cv.subjects)
        for key in required_keys:
            self.assertEqual(len(results[key]), n_folds)
            self.assertTrue(np.all(np.isfinite(results[key])))


if __name__ == "__main__":
    unittest.main()
