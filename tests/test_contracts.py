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
    compressed: bool = False,
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

    if compressed:
        np.savez_compressed(data_dir / "X_pre.npz", data=X)
        np.savez_compressed(data_dir / "y_heater.npz", data=y)
        np.savez_compressed(data_dir / "subjects.npz", data=subjects)
    else:
        np.save(data_dir / "X_pre.npy", X)
        np.save(data_dir / "y_heater.npy", y)
        np.save(data_dir / "subjects.npy", subjects)


def _write_synthetic_biovid_dataset(
    data_dir: Path,
    num_subjects: int = 4,
    num_classes: int = 5,
    train_samples_per_class: int = 6,
    test_samples_per_class: int = 2,
    seq_len: int = 1152,
    compressed: bool = False,
) -> None:
    """Create a tiny BioVid-like PartA train/test tree for split-contract tests."""
    rng = np.random.default_rng(456)
    root = data_dir / "BioVid" / "PartA"
    modalities = ("GSR", "ECG", "EMG")
    splits = {
        "Train": train_samples_per_class,
        "Test": test_samples_per_class,
    }
    subject_keys = [f"subject_{idx:02d}" for idx in range(num_subjects)]

    for split_name, samples_per_class in splits.items():
        for modality in modalities:
            (root / split_name / modality).mkdir(parents=True, exist_ok=True)

        for subject_idx, subject_key in enumerate(subject_keys):
            labels = np.concatenate(
                [
                    np.full((samples_per_class, 1), class_id, dtype=np.uint8)
                    for class_id in range(num_classes)
                ],
                axis=0,
            )
            for modality_idx, modality in enumerate(modalities):
                # Inject small modality-specific shifts so the channels are not identical.
                mean_shift = float(subject_idx * 0.25 + modality_idx * 0.1)
                data = np.concatenate(
                    [
                        rng.normal(
                            loc=class_id + mean_shift,
                            scale=0.5,
                            size=(samples_per_class, seq_len, 1),
                        ).astype(np.float32)
                        for class_id in range(num_classes)
                    ],
                    axis=0,
                )
                if compressed:
                    np.savez_compressed(
                        root / split_name / modality / f"{subject_key}_data.npz",
                        data=data,
                    )
                    np.savez_compressed(
                        root / split_name / modality / f"{subject_key}_label.npz",
                        data=labels,
                    )
                else:
                    np.save(
                        root / split_name / modality / f"{subject_key}_data.npy",
                        data,
                    )
                    np.save(
                        root / split_name / modality / f"{subject_key}_label.npy",
                        labels,
                    )


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)
        _write_synthetic_dataset(self.data_dir)

        self.config = PainDatasetConfig(
            dataset_source="painmonit",
            sequence_length=2500,
            n_way=6,
            num_stimuli_levels=6,
            k_shot=1,
            q_query=1,
            num_epochs=1,
            tasks_per_epoch=1,
            val_tasks=1,
            heldout_eval_tasks=1,
            single_loso_fold=False,
            seed=7,
            deterministic_ops=True,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_loso_split_contracts(self):
        dataset = PainMetaDataset(data_dir=str(self.data_dir), config=self.config)
        cv = LOSOCrossValidator(
            dataset=dataset,
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

            support_counts = np.bincount(task["support_y"], minlength=self.config.n_way)
            query_counts = np.bincount(task["query_y"], minlength=self.config.n_way)
            self.assertTrue(np.all(support_counts == self.config.k_shot))
            self.assertTrue(np.all(query_counts == self.config.q_query))

        expected_train_subject = (
            -1 if len(fold["train_subjects"]) > 1 else fold["train_subjects"][0]
        )
        expected_val_subject = (
            -1 if len(fold["val_subjects"]) > 1 else fold["val_subjects"][0]
        )
        self.assertEqual(
            fold["train_sampler"].get_task()["subject"], expected_train_subject
        )
        self.assertEqual(
            fold["val_sampler"].get_task()["subject"], expected_val_subject
        )
        self.assertEqual(
            fold["test_sampler"].get_task()["subject"],
            int(dataset.unique_subjects[0]),
        )

    def test_test_sampler_uses_available_query_budget(self):
        tmp = tempfile.TemporaryDirectory()
        data_dir = Path(tmp.name)
        _write_synthetic_dataset(data_dir, num_subjects=4)
        config = PainDatasetConfig(
            dataset_source="painmonit",
            sequence_length=2500,
            n_way=6,
            num_stimuli_levels=6,
            k_shot=3,
            q_query=2,
            num_epochs=1,
            tasks_per_epoch=1,
            val_tasks=1,
            heldout_eval_tasks=1,
            single_loso_fold=False,
            seed=7,
            deterministic_ops=True,
        )
        try:
            dataset = PainMetaDataset(data_dir=str(data_dir), config=config)
            cv = LOSOCrossValidator(dataset=dataset, seed=config.seed)
            fold = cv.get_fold(test_subject=int(dataset.unique_subjects[0]))

            train_task = fold["train_sampler"].get_task()
            train_support_counts = np.bincount(
                train_task["support_y"], minlength=config.n_way
            )
            train_query_counts = np.bincount(
                train_task["query_y"], minlength=config.n_way
            )
            self.assertEqual(train_task["subject"], -1)
            self.assertTrue(np.all(train_support_counts == config.k_shot))
            self.assertTrue(np.all(train_query_counts == config.q_query))

            test_task = fold["test_sampler"].get_task()
            test_support_counts = np.bincount(
                test_task["support_y"], minlength=config.n_way
            )
            test_query_counts = np.bincount(
                test_task["query_y"], minlength=config.n_way
            )
            self.assertTrue(np.all(test_support_counts == config.k_shot))
            self.assertTrue(np.all(test_query_counts == 1))
        finally:
            tmp.cleanup()

    def test_training_contracts_smoke(self):
        learner = FewShotPainLearner(
            config=self.config,
            data_dir=str(self.data_dir),
            learning_rate=1e-3,
            fusion_method="mean",
        )

        results = learner.train()

        required_keys = {
            "train_losses",
            "train_accuracies",
            "val_losses",
            "val_accuracies",
            "test_losses",
            "test_accuracies",
            "zero_shot_intra_class_similarities",
            "zero_shot_inter_class_similarities",
            "k_shot_intra_class_similarities",
            "k_shot_inter_class_similarities",
        }
        self.assertTrue(required_keys.issubset(results.keys()))

        n_folds = len(learner.cv.subjects)
        for key in required_keys:
            self.assertEqual(len(results[key]), n_folds)
            self.assertTrue(np.all(np.isfinite(results[key])))

    def test_two_way_task_class_mapping(self):
        config = PainDatasetConfig(
            dataset_source="painmonit",
            sequence_length=2500,
            num_stimuli_levels=6,
            task_class_ids=(0, 5),
            k_shot=1,
            q_query=1,
            num_epochs=1,
            tasks_per_epoch=1,
            val_tasks=1,
            heldout_eval_tasks=1,
            single_loso_fold=False,
            seed=7,
            deterministic_ops=True,
        )
        self.assertEqual(config.n_way, 2)

        dataset = PainMetaDataset(data_dir=str(self.data_dir), config=config)
        cv = LOSOCrossValidator(dataset=dataset, seed=config.seed)
        fold = cv.get_fold(test_subject=int(dataset.unique_subjects[0]))
        task = fold["test_sampler"].get_task()

        self.assertEqual(task["support_X"].shape[0], 2 * config.k_shot)
        self.assertEqual(task["query_X"].shape[0], 2 * config.q_query)
        self.assertTrue(np.array_equal(np.unique(task["support_y"]), np.array([0, 1])))
        self.assertTrue(np.array_equal(np.unique(task["query_y"]), np.array([0, 1])))

    def test_biovid_predefined_split_contracts(self):
        tmp = tempfile.TemporaryDirectory()
        data_dir = Path(tmp.name)
        _write_synthetic_biovid_dataset(data_dir)

        config = PainDatasetConfig(
            dataset_source="biovid_part_a",
            k_shot=1,
            q_query=1,
            num_epochs=1,
            tasks_per_epoch=1,
            val_tasks=1,
            heldout_eval_tasks=1,
            single_loso_fold=False,
            seed=13,
            deterministic_ops=True,
        )
        try:
            # BioVid default binary classes should adapt to (0, 4).
            self.assertEqual(config.task_class_ids, (0, 4))
            self.assertEqual(config.split_strategy, "predefined")
            self.assertFalse(config.enable_window_shift_augmentation)

            dataset = PainMetaDataset(data_dir=str(data_dir), config=config)
            self.assertTrue(dataset.has_predefined_split)
            self.assertEqual(
                sorted(dataset.get_split_subjects("train")),
                sorted(dataset.get_split_subjects("test")),
            )

            train_index = dataset._get_index_for_split("train")
            test_index = dataset._get_index_for_split("test")
            for subject in dataset.get_split_subjects("train"):
                for class_id in range(config.n_way):
                    self.assertGreater(
                        len(train_index[int(subject)][class_id]),
                        len(test_index[int(subject)][class_id]),
                    )

            cv = LOSOCrossValidator(dataset=dataset, seed=config.seed)
            test_subjects = sorted(dataset.get_split_subjects("test"))
            self.assertEqual(sorted(cv.subjects), test_subjects)

            held_out_subject = int(test_subjects[1])
            fold = cv.get_fold(test_subject=held_out_subject)

            self.assertEqual(fold["test_subject"], held_out_subject)
            self.assertEqual(fold["test_subjects"], [held_out_subject])
            self.assertNotIn(held_out_subject, fold["train_subjects"])
            self.assertNotIn(held_out_subject, fold["val_subjects"])
            self.assertEqual(
                sorted(fold["val_subjects"]),
                [subject for subject in test_subjects if subject != held_out_subject],
            )

            train_task = fold["train_sampler"].get_task()
            val_task = fold["val_sampler"].get_task()
            test_task = fold["test_sampler"].get_task()
            expected_support = config.n_way * config.k_shot
            expected_query = config.n_way * config.q_query
            self.assertEqual(train_task["support_X"].shape, (expected_support, 1152, 3))
            self.assertEqual(train_task["query_X"].shape, (expected_query, 1152, 3))
            self.assertEqual(val_task["support_X"].shape, (expected_support, 1152, 3))
            self.assertEqual(val_task["query_X"].shape, (expected_query, 1152, 3))
            self.assertEqual(test_task["support_X"].shape, (expected_support, 1152, 3))
            self.assertEqual(test_task["query_X"].shape, (expected_query, 1152, 3))
            self.assertNotEqual(train_task["subject"], held_out_subject)
            self.assertNotEqual(val_task["subject"], held_out_subject)
            self.assertEqual(test_task["subject"], held_out_subject)
        finally:
            tmp.cleanup()

    def test_painmonit_loader_accepts_compressed_npz_arrays(self):
        tmp = tempfile.TemporaryDirectory()
        data_dir = Path(tmp.name)
        _write_synthetic_dataset(data_dir, compressed=True)

        config = PainDatasetConfig(
            dataset_source="painmonit",
            sequence_length=2500,
            n_way=6,
            num_stimuli_levels=6,
            k_shot=1,
            q_query=1,
            num_epochs=1,
            tasks_per_epoch=1,
            val_tasks=1,
            heldout_eval_tasks=1,
            single_loso_fold=False,
            seed=7,
            deterministic_ops=True,
        )
        try:
            dataset = PainMetaDataset(data_dir=str(data_dir), config=config)
            self.assertEqual(dataset.X.shape, (72, 2500, 3))
            self.assertEqual(dataset.y.shape, (72,))
            self.assertEqual(dataset.subjects.shape, (72,))
        finally:
            tmp.cleanup()

    def test_biovid_loader_accepts_compressed_npz_tree(self):
        tmp = tempfile.TemporaryDirectory()
        data_dir = Path(tmp.name)
        _write_synthetic_biovid_dataset(data_dir, compressed=True)

        config = PainDatasetConfig(
            dataset_source="biovid_part_a",
            k_shot=1,
            q_query=1,
            num_epochs=1,
            tasks_per_epoch=1,
            val_tasks=1,
            heldout_eval_tasks=1,
            single_loso_fold=False,
            seed=13,
            deterministic_ops=True,
        )
        try:
            dataset = PainMetaDataset(data_dir=str(data_dir), config=config)
            self.assertTrue(dataset.has_predefined_split)
            self.assertEqual(dataset.X.shape[1:], (1152, 3))
            self.assertEqual(sorted(np.unique(dataset.y).tolist()), [0, 1, 2, 3, 4])
            self.assertEqual(
                sorted(dataset.get_split_subjects("train")),
                sorted(dataset.get_split_subjects("test")),
            )
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
