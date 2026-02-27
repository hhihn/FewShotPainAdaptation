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
        k_shot: int = 5,
        q_query: int = 5,
        tasks_per_epoch: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Initialize cross-validator.

        Args:
            dataset: PainMetaDataset instance
            k_shot: Support set size per class
            q_query: Query set size per class
            tasks_per_epoch: Tasks per training epoch
            seed: Random seed
        """
        self.dataset = dataset
        self.k_shot = k_shot
        self.q_query = q_query
        self.tasks_per_epoch = tasks_per_epoch
        self.seed = seed
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
        train_subjects, _ = self.dataset.leave_one_subject_out_split(test_subject)
        fold_seed = None if self.seed is None else int(self.seed) + int(test_subject)
        val_seed = None if fold_seed is None else fold_seed + 10_000
        test_seed = None if fold_seed is None else fold_seed + 20_000
        self.logger.debug(f"Train subjects: {train_subjects}")
        # Split training subjects into train and validation
        n_val = max(1, len(train_subjects) // 10)  # 10% for validation
        self.logger.debug(f"n_val: {n_val}")
        val_subjects = train_subjects[-n_val:]
        self.logger.debug(f"val_subjects: {val_subjects}")
        train_subjects_final = train_subjects[:-n_val]
        self.logger.debug(f"train_subjects_final: {train_subjects_final}")

        train_sampler = SixWayKShotSampler(
            dataset=self.dataset,
            mode="train",
            train_subjects=train_subjects_final,
            test_subject=test_subject,
            k_shot=self.k_shot,
            q_query=self.q_query,
            tasks_per_epoch=self.tasks_per_epoch,
            seed=fold_seed,
        )

        val_sampler = SixWayKShotSampler(
            dataset=self.dataset,
            mode="val",
            train_subjects=val_subjects,
            test_subject=test_subject,
            k_shot=self.k_shot,
            q_query=self.q_query,
            tasks_per_epoch=self.tasks_per_epoch // 5,
            seed=val_seed,
        )

        test_sampler = SixWayKShotSampler(
            dataset=self.dataset,
            mode="test",
            train_subjects=train_subjects,
            test_subject=test_subject,
            k_shot=self.k_shot,
            q_query=self.q_query,
            tasks_per_epoch=20,  # Fewer tasks for testing
            seed=test_seed,
        )

        return {
            "train_sampler": train_sampler,
            "val_sampler": val_sampler,
            "test_sampler": test_sampler,
            "test_subject": test_subject,
            "train_subjects": train_subjects_final,
            "val_subjects": val_subjects,
            "n_train_subjects": len(train_subjects_final),
            "n_val_subjects": len(val_subjects),
        }
