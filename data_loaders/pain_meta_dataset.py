#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
N-Way-K-Shot Meta-Learning Sampler for Multi-Modal Pain Dataset.

This sampler creates meta-learning tasks from the BioVid pain dataset
with the following structure:
- N-way: configurable pain/temperature levels (classes)
- K-shot: K support samples per class
- Q-query: Q query samples per class for evaluation

The sampler supports leave-one-subject-out cross-validation.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import warnings

from utils.logger import setup_logger
from data_loaders.pain_ds_config import PainDatasetConfig


class PainMetaDataset:
    """
    Meta-learning dataset for multi-modal pain assessment.

    Handles the BioVid-style pain dataset with:
    - 52 subjects (51 after excluding corrupted data)
    - 6 pain/temperature levels
    - 8 repetitions per level
    - 6 sensor modalities

    Provides episodic sampling for meta-learning with N-way-K-shot tasks.
    """

    def __init__(
        self,
        data_dir: str,
        config: Optional[PainDatasetConfig] = None,
        normalize: bool = True,
        normalize_per_subject: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing X.npy, y_heater.npy, subjects.npy
            config: Dataset configuration
            normalize: Whether to normalize the data
            normalize_per_subject: If True, normalize per subject; else global normalization
        """
        self.logger = setup_logger("PainMetaDataset")
        self.config = config or PainDatasetConfig()
        self.task_class_ids = tuple(int(class_id) for class_id in self.config.task_class_ids)
        self.data_dir = Path(data_dir)
        self.logger.debug(f"Data directory: {self.data_dir}")
        self.normalize = normalize
        self.normalize_per_subject = normalize_per_subject

        # Load data
        self._load_data()

        # Build index for efficient sampling
        self._build_index()

        # Compute normalization statistics
        if self.normalize:
            self._compute_normalization_stats()

    def _load_data(self):
        """Load data arrays from disk."""
        self.logger.info(f"Loading data from {self.data_dir}...")

        # Load arrays
        self.X = np.load(self.data_dir / self.config.data_path)[
            :, :, self.config.sensor_idx, :
        ]
        self.logger.info(f"X.shape: {self.X.shape}")
        self.y_onehot = np.load(self.data_dir / self.config.labels_path)
        self.logger.info(f"y_onehot.shape: {self.y_onehot.shape}")
        self.subjects = np.load(self.data_dir / self.config.subjects_path)
        self.logger.info(f"subjects.shape: {self.subjects}")

        # Convert one-hot to class indices
        self.y = np.argmax(self.y_onehot, axis=1)

        # Remove the trailing dimension if present (for CNN compatibility)
        if self.X.ndim == 4 and self.X.shape[-1] == 1:
            self.X = np.squeeze(self.X, axis=-1)

        # Get unique subjects
        self.unique_subjects = np.unique(self.subjects)
        self.logger.debug(f"Unique subjects: {self.unique_subjects}")
        self.num_subjects = len(self.unique_subjects)
        self.logger.info(f"Number of subjects: {self.num_subjects}")
        self.logger.info(f"  Data shape: {self.X.shape}")
        self.logger.info(f"  Labels shape: {self.y.shape}")
        self.logger.info(f"  Number of subjects: {self.num_subjects}")
        self.logger.info(f"  Samples per subject: ~{len(self.y) // self.num_subjects}")
        self.logger.info(f"  Classes: {np.unique(self.y)}")
        self.logger.info(f"  Task classes: {self.task_class_ids}")

    def _build_index(self):
        """Build index mapping (subject, class) -> sample indices."""
        self.index = {}

        for subject in self.unique_subjects:
            self.index[subject] = {}
            subject_mask = self.subjects == subject

            for episodic_class_id, raw_class_id in enumerate(self.task_class_ids):
                class_mask = self.y == raw_class_id
                combined_mask = subject_mask & class_mask
                indices = np.where(combined_mask)[0]
                self.index[subject][episodic_class_id] = indices

        # Verify index
        self._verify_index()

    def _verify_index(self):
        """Verify that the index is valid for sampling."""
        min_samples_per_class = float("inf")

        for subject in self.unique_subjects:
            for episodic_class_id, raw_class_id in enumerate(self.task_class_ids):
                n_samples = len(self.index[subject][episodic_class_id])
                min_samples_per_class = min(min_samples_per_class, n_samples)

                if n_samples < self.config.k_shot + self.config.q_query:
                    warnings.warn(
                        f"Subject {subject}, raw class {raw_class_id} has only {n_samples} samples, "
                        f"but {self.config.k_shot + self.config.q_query} are needed for sampling."
                    )

        self.logger.info(
            f"  Minimum samples per (subject, class): {min_samples_per_class}"
        )

    def _compute_normalization_stats(self):
        """Compute mean and std for normalization."""
        if self.normalize_per_subject:
            # Per-subject normalization
            self.norm_stats = {}
            for subject in self.unique_subjects:
                subject_mask = self.subjects == subject
                subject_data = self.X[subject_mask]
                self.norm_stats[subject] = {
                    "mean": np.mean(subject_data, axis=(0, 1), keepdims=True),
                    "std": np.std(subject_data, axis=(0, 1), keepdims=True) + 1e-8,
                }
        else:
            # Global normalization
            self.global_mean = np.mean(self.X, axis=(0, 1), keepdims=True)
            self.global_std = np.std(self.X, axis=(0, 1), keepdims=True) + 1e-8

    def _normalize_data(
        self, data: np.ndarray, subject: Optional[int] = None
    ) -> np.ndarray:
        """Normalize data."""
        if not self.normalize:
            return data

        if self.normalize_per_subject and subject is not None:
            mean = self.norm_stats[subject]["mean"]
            std = self.norm_stats[subject]["std"]
        else:
            mean = self.global_mean
            std = self.global_std

        return (data - mean) / std

    def _normalize_data_by_subjects(
        self, data: np.ndarray, subjects: np.ndarray
    ) -> np.ndarray:
        """Normalize each sample using the statistics of its source subject."""
        if not self.normalize or data.size == 0:
            return data

        if not self.normalize_per_subject:
            return self._normalize_data(data)

        normalized = data.copy()
        for subject in np.unique(subjects):
            subject_mask = subjects == subject
            normalized[subject_mask] = self._normalize_data(
                normalized[subject_mask],
                int(subject),
            )
        return normalized

    @staticmethod
    def _resolve_total_samples_to_draw(
        available_count: int,
        k_shot: int,
        q_query: int,
        allow_partial_query: bool,
    ) -> int:
        """Resolve how many samples may be drawn while preserving k-shot support."""
        requested_total = k_shot + q_query
        if available_count < k_shot:
            raise ValueError(
                f"Only {available_count} samples available, but at least {k_shot} are required for support."
            )
        if not allow_partial_query and available_count < requested_total:
            raise ValueError(
                f"Only {available_count} samples available, but {requested_total} are required."
            )
        return max(min(available_count, requested_total), k_shot)

    @staticmethod
    def _compute_batch_stats(data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute mean/std stats from a batch [n, seq_len, n_sensors]."""
        return {
            "mean": np.mean(data, axis=(0, 1), keepdims=True),
            "std": np.std(data, axis=(0, 1), keepdims=True) + 1e-8,
        }

    @staticmethod
    def _apply_stats(data: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """Normalize a batch with externally provided stats."""
        return (data - stats["mean"]) / stats["std"]

    def get_subject_data(
        self, subject: int, normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all data for a specific subject.

        Args:
            subject: Subject ID
            normalize: Whether to apply normalization

        Returns:
            X: Data array of shape [n_samples, sequence_length, n_sensors]
            y: Labels array of shape [n_samples]
        """
        mask = self.subjects == subject
        X = self.X[mask].copy()
        y = self.y[mask].copy()

        if normalize:
            X = self._normalize_data(X, subject)

        return X, y

    def sample_task(
        self,
        subject: int,
        k_shot: Optional[int] = None,
        q_query: Optional[int] = None,
        seed: Optional[int] = None,
        normalize_mode: str = "subject",
        rng: Optional[np.random.Generator] = None,
        allow_partial_query: bool = False,
        include_sample_subjects: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Sample an N-way-K-shot task from a single subject.

        Args:
            subject: Subject ID to sample from
            k_shot: Number of support samples per class (default: config.k_shot)
            q_query: Number of query samples per class (default: config.q_query)
            seed: Random seed for reproducibility
            normalize_mode: One of:
                - 'subject': normalize with precomputed per-subject/global stats
                - 'support': normalize both support/query using support-set stats only
                - 'none': no normalization
            rng: Optional numpy Generator to control sampling deterministically
            allow_partial_query: If True, keep k-shot support and use all remaining
                samples for query when a subject has fewer than k_shot + q_query items

        Returns:
            Dictionary containing:
                - support_X: [n_way * k_shot, seq_len, n_sensors]
                - support_y: [n_way * k_shot]
                - query_X: [n_way * q_query, seq_len, n_sensors]
                - query_y: [n_way * q_query]
        """
        return self.sample_task_from_subjects(
            subjects=[subject],
            k_shot=k_shot,
            q_query=q_query,
            seed=seed,
            normalize_mode=normalize_mode,
            rng=rng,
            allow_partial_query=allow_partial_query,
            include_sample_subjects=include_sample_subjects,
        )

    def sample_task_from_subjects(
        self,
        subjects: List[int],
        k_shot: Optional[int] = None,
        q_query: Optional[int] = None,
        seed: Optional[int] = None,
        normalize_mode: str = "subject",
        rng: Optional[np.random.Generator] = None,
        allow_partial_query: bool = False,
        include_sample_subjects: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Sample one task by pooling each class across the provided subjects."""
        k_shot = k_shot or self.config.k_shot
        q_query = q_query or self.config.q_query
        selected_subjects = [int(subject) for subject in subjects]
        if not selected_subjects:
            raise ValueError("subjects must contain at least one subject id")

        if rng is not None:
            local_rng = rng
        elif seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = np.random.default_rng()

        support_X, support_y, support_subjects = [], [], []
        query_X, query_y, query_subjects = [], [], []

        for class_id in range(self.config.n_way):
            pooled_indices = []
            pooled_subject_ids = []
            for subject in selected_subjects:
                indices = self.index[subject][class_id]
                pooled_indices.extend(indices.tolist())
                pooled_subject_ids.extend([subject] * len(indices))

            total_to_sample = self._resolve_total_samples_to_draw(
                available_count=len(pooled_indices),
                k_shot=k_shot,
                q_query=q_query,
                allow_partial_query=allow_partial_query,
            )
            sampled_positions = local_rng.choice(
                len(pooled_indices), size=total_to_sample, replace=False
            )
            pooled_indices = np.asarray(pooled_indices, dtype=np.int64)
            pooled_subject_ids = np.asarray(pooled_subject_ids, dtype=np.int32)

            sampled_indices = pooled_indices[sampled_positions]
            sampled_subject_ids = pooled_subject_ids[sampled_positions]
            support_idx = sampled_indices[:k_shot]
            query_idx = sampled_indices[k_shot:]
            support_subject_ids = sampled_subject_ids[:k_shot]
            query_subject_ids = sampled_subject_ids[k_shot:]

            support_X.append(self.X[support_idx])
            support_y.append(np.full(k_shot, class_id, dtype=np.int32))
            support_subjects.append(support_subject_ids)
            query_X.append(self.X[query_idx])
            query_y.append(np.full(len(query_idx), class_id, dtype=np.int32))
            query_subjects.append(query_subject_ids)

        support_X = np.concatenate(support_X, axis=0)
        support_y = np.concatenate(support_y, axis=0)
        support_subjects = np.concatenate(support_subjects, axis=0)
        query_X = np.concatenate(query_X, axis=0)
        query_y = np.concatenate(query_y, axis=0)
        query_subjects = np.concatenate(query_subjects, axis=0)

        if normalize_mode == "subject":
            support_X = self._normalize_data_by_subjects(support_X, support_subjects)
            query_X = self._normalize_data_by_subjects(query_X, query_subjects)
        elif normalize_mode == "support":
            stats = self._compute_batch_stats(support_X)
            support_X = self._apply_stats(support_X, stats)
            query_X = self._apply_stats(query_X, stats)
        elif normalize_mode == "none":
            pass
        else:
            raise ValueError(
                f"Unknown normalize_mode: {normalize_mode}. Use 'subject', 'support', or 'none'."
            )

        support_perm = local_rng.permutation(len(support_y))
        query_perm = local_rng.permutation(len(query_y))
        task_subject = selected_subjects[0] if len(selected_subjects) == 1 else -1

        task = {
            "support_X": support_X[support_perm],
            "support_y": support_y[support_perm],
            "query_X": query_X[query_perm],
            "query_y": query_y[query_perm],
            "subject": task_subject,
        }
        if include_sample_subjects:
            task["support_subjects"] = support_subjects[support_perm]
            task["query_subjects"] = query_subjects[query_perm]
        return task

    def sample_task_cross_subjects(
        self,
        support_subject: int,
        query_subject: int,
        k_shot: Optional[int] = None,
        q_query: Optional[int] = None,
        seed: Optional[int] = None,
        normalize_mode: str = "subject",
        rng: Optional[np.random.Generator] = None,
        include_sample_subjects: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Sample one task with support drawn from one subject and query from another.

        Args:
            support_subject: Subject used for all support samples
            query_subject: Subject used for all query samples
            k_shot: Number of support samples per class
            q_query: Number of query samples per class
            seed: Random seed for reproducibility
            normalize_mode: One of 'subject', 'support', or 'none'
            rng: Optional numpy Generator to control sampling deterministically
            include_sample_subjects: Whether to return support/query subject ids

        Returns:
            Dictionary containing support/query arrays and labels.
        """
        k_shot = k_shot or self.config.k_shot
        q_query = q_query or self.config.q_query
        support_subject = int(support_subject)
        query_subject = int(query_subject)

        if rng is not None:
            local_rng = rng
        elif seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = np.random.default_rng()

        support_X, support_y, support_subjects = [], [], []
        query_X, query_y, query_subjects = [], [], []

        for class_id in range(self.config.n_way):
            support_indices = self.index[support_subject][class_id]
            query_indices = self.index[query_subject][class_id]

            if len(support_indices) < k_shot:
                raise ValueError(
                    f"Support subject {support_subject} has only {len(support_indices)} "
                    f"samples for class {class_id}, but k_shot={k_shot} was requested."
                )
            if len(query_indices) < q_query:
                raise ValueError(
                    f"Query subject {query_subject} has only {len(query_indices)} "
                    f"samples for class {class_id}, but q_query={q_query} was requested."
                )

            sampled_support_idx = local_rng.choice(
                support_indices, size=k_shot, replace=False
            )
            sampled_query_idx = local_rng.choice(
                query_indices, size=q_query, replace=False
            )

            support_X.append(self.X[sampled_support_idx])
            support_y.append(np.full(k_shot, class_id, dtype=np.int32))
            support_subjects.append(np.full(k_shot, support_subject, dtype=np.int32))
            query_X.append(self.X[sampled_query_idx])
            query_y.append(np.full(q_query, class_id, dtype=np.int32))
            query_subjects.append(np.full(q_query, query_subject, dtype=np.int32))

        support_X = np.concatenate(support_X, axis=0)
        support_y = np.concatenate(support_y, axis=0)
        support_subjects = np.concatenate(support_subjects, axis=0)
        query_X = np.concatenate(query_X, axis=0)
        query_y = np.concatenate(query_y, axis=0)
        query_subjects = np.concatenate(query_subjects, axis=0)

        if normalize_mode == "subject":
            support_X = self._normalize_data_by_subjects(support_X, support_subjects)
            query_X = self._normalize_data_by_subjects(query_X, query_subjects)
        elif normalize_mode == "support":
            stats = self._compute_batch_stats(support_X)
            support_X = self._apply_stats(support_X, stats)
            query_X = self._apply_stats(query_X, stats)
        elif normalize_mode == "none":
            pass
        else:
            raise ValueError(
                f"Unknown normalize_mode: {normalize_mode}. Use 'subject', 'support', or 'none'."
            )

        support_perm = local_rng.permutation(len(support_y))
        query_perm = local_rng.permutation(len(query_y))
        task = {
            "support_X": support_X[support_perm],
            "support_y": support_y[support_perm],
            "query_X": query_X[query_perm],
            "query_y": query_y[query_perm],
            "subject": -1,
        }
        if include_sample_subjects:
            task["support_subjects"] = support_subjects[support_perm]
            task["query_subjects"] = query_subjects[query_perm]
        return task

    def sample_meta_task_batch(
        self,
        subjects: List[int],
        batch_size: int,
        k_shot: Optional[int] = None,
        q_query: Optional[int] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Sample a batch of tasks for meta-training.

        Args:
            subjects: List of subject IDs to sample from
            batch_size: Number of tasks to sample
            k_shot: Support set size per class
            q_query: Query set size per class

        Returns:
            List of task dictionaries
        """
        sampled_subjects = np.random.choice(subjects, size=batch_size, replace=True)
        return [self.sample_task(s, k_shot, q_query) for s in sampled_subjects]

    def leave_one_subject_out_split(self, test_subject: int) -> Tuple[List[int], int]:
        """
        Create leave-one-subject-out split.

        Args:
            test_subject: Subject ID to hold out for testing

        Returns:
            train_subjects: List of training subject IDs
            test_subject: Held-out subject ID
        """
        train_subjects = [s for s in self.unique_subjects if s != test_subject]
        return train_subjects, test_subject

    def get_few_shot_split(
        self, subject: int, k_shot: int, seed: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Get a few-shot split for adaptation and evaluation.

        Args:
            subject: Subject ID
            k_shot: Number of shots (support samples per class)
            seed: Random seed

        Returns:
            support_set: Dictionary with support data
            eval_set: Dictionary with remaining data for evaluation
        """
        local_rng = np.random.default_rng(seed)

        support_X, support_y = [], []
        eval_X, eval_y = [], []

        for class_id in range(self.config.n_way):
            indices = self.index[subject][class_id]
            shuffled_indices = local_rng.permutation(indices)

            support_idx = shuffled_indices[:k_shot]
            eval_idx = shuffled_indices[k_shot:]

            support_X.append(self.X[support_idx])
            support_y.append(np.full(len(support_idx), class_id, dtype=np.int32))
            eval_X.append(self.X[eval_idx])
            eval_y.append(np.full(len(eval_idx), class_id, dtype=np.int32))

        support_X = np.concatenate(support_X, axis=0)
        support_y = np.concatenate(support_y, axis=0)
        eval_X = np.concatenate(eval_X, axis=0)
        eval_y = np.concatenate(eval_y, axis=0)

        # Normalize
        support_X = self._normalize_data(support_X, subject)
        eval_X = self._normalize_data(eval_X, subject)

        support_set = {"X": support_X, "y": support_y}
        eval_set = {"X": eval_X, "y": eval_y}

        return support_set, eval_set
