from typing import Dict, Optional, Generator

from utils.logger import setup_logger

from data_loaders.pain_meta_dataset import PainMetaDataset
from data_loaders.meta_ds_sampler import SixWayKShotSampler


class LOSOCrossValidator:
    """
    Leave-One-Subject-Out Cross-Validator for meta-learning.

    Provides an iterator over all LOSO folds, where each fold uses
    one subject for testing and the rest for training.
    """

    def __init__(
        self,
        dataset: PainMetaDataset,
        seed: Optional[int] = None,
    ):
        """
        Initialize cross-validator.

        Args:
            dataset: PainMetaDataset instance
            seed: Random seed
        """
        self.dataset = dataset
        self.config = dataset.config
        self.k_shot = self.config.k_shot
        self.q_query = self.config.q_query
        self.tasks_per_epoch = self.config.tasks_per_epoch
        self.seed = seed
        if dataset.has_predefined_split:
            # Fixed train/test evaluation uses a single fold.
            self.subjects = [-1]
        else:
            self.subjects = list(dataset.unique_subjects)

        self.logger = setup_logger(__name__)

    def __len__(self) -> int:
        """Number of folds (one per subject)."""
        return len(self.subjects)

    def __iter__(self) -> Generator[Dict[str, any], None, None]:
        """Iterate over LOSO folds."""
        for test_subject in self.subjects:
            yield self.get_fold(test_subject)

    def get_fold(self, test_subject: int) -> Dict[str, any]:
        """
        Get samplers for a specific fold.

        Args:
            test_subject: Subject ID for testing

        Returns:
            Dictionary with train_sampler, val_sampler, test_sampler, and metadata
        """
        if self.dataset.has_predefined_split:
            train_subjects = self.dataset.get_split_subjects("train")
            test_subjects = self.dataset.get_split_subjects("test")
            effective_test_subject = -1
            fold_seed = None if self.seed is None else int(self.seed)
            train_data_split = "train"
            test_data_split = "test"
        else:
            train_subjects, _ = self.dataset.leave_one_subject_out_split(test_subject)
            test_subjects = [int(test_subject)]
            effective_test_subject = int(test_subject)
            fold_seed = None if self.seed is None else int(self.seed) + int(test_subject)
            train_data_split = "all"
            test_data_split = "all"

        val_seed = None if fold_seed is None else fold_seed + 10_000
        test_seed = None if fold_seed is None else fold_seed + 20_000
        self.logger.debug(f"Train subjects: {train_subjects}")
        # Split training subjects into train and validation.
        if len(train_subjects) <= 1:
            val_subjects = list(train_subjects)
            train_subjects_final = list(train_subjects)
        else:
            n_val = max(1, len(train_subjects) // 10)  # 10% for validation
            self.logger.debug(f"n_val: {n_val}")
            val_subjects = train_subjects[-n_val:]
            train_subjects_final = train_subjects[:-n_val]
            if not train_subjects_final:
                train_subjects_final = list(train_subjects)
        self.logger.debug(f"val_subjects: {val_subjects}")
        self.logger.debug(f"train_subjects_final: {train_subjects_final}")
        if self.dataset.window_shift_enabled:
            self.dataset.log_window_shift_split_summary(
                "train",
                train_subjects_final,
                split=train_data_split,
            )
            self.dataset.log_window_shift_split_summary(
                "val",
                val_subjects,
                split=train_data_split,
            )
            self.dataset.log_window_shift_split_summary(
                "test",
                test_subjects,
                split=test_data_split,
            )

        train_sampler = SixWayKShotSampler(
            dataset=self.dataset,
            mode="train",
            train_subjects=train_subjects_final,
            test_subject=effective_test_subject,
            seed=fold_seed,
            data_split=train_data_split,
        )

        val_sampler = SixWayKShotSampler(
            dataset=self.dataset,
            mode="val",
            train_subjects=val_subjects,
            test_subject=effective_test_subject,
            seed=val_seed,
            data_split=train_data_split,
        )

        test_sampler = SixWayKShotSampler(
            dataset=self.dataset,
            mode="test",
            train_subjects=train_subjects,
            test_subject=effective_test_subject,
            test_subjects=test_subjects,
            seed=test_seed,
            data_split=test_data_split,
        )

        return {
            "train_sampler": train_sampler,
            "val_sampler": val_sampler,
            "test_sampler": test_sampler,
            "test_subject": effective_test_subject,
            "test_subjects": test_subjects,
            "train_subjects": train_subjects_final,
            "val_subjects": val_subjects,
            "n_train_subjects": len(train_subjects_final),
            "n_val_subjects": len(val_subjects),
        }
