import numpy as np
import tensorflow as tf
from typing import List, Dict, Optional, Generator


from utils.logger import setup_logger

from data_loaders.pain_meta_dataset import PainMetaDataset


class SixWayKShotSampler:
    """
    6-Way-K-Shot episodic sampler for meta-learning.

    This sampler generates tasks for training and evaluating meta-learning
    models on the pain dataset. Each task contains:
    - Support set: K samples from each of the 6 pain levels
    - Query set: Q samples from each of the 6 pain levels

    Supports:
    - Leave-one-subject-out cross-validation
    - Configurable K-shot and Q-query sizes
    - Multi-modal data handling
    - TensorFlow Dataset integration
    """

    def __init__(
        self,
        dataset: PainMetaDataset,
        mode: str = "train",
        train_subjects: Optional[List[int]] = None,
        test_subject: Optional[int] = None,
        test_subjects: Optional[List[int]] = None,
        data_split: str = "all",
        seed: Optional[int] = None,
    ):
        """
        Initialize the sampler.

        Args:
            dataset: PainMetaDataset instance
            mode: 'train', 'val', or 'test'
            train_subjects: List of training subject IDs
            test_subject: Held-out test subject ID
            seed: Random seed
        """
        self.logger = setup_logger(__name__)
        self.dataset = dataset
        self.mode = mode
        self.config = dataset.config
        self.k_shot = self.config.k_shot
        self.q_query = self.config.q_query
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.data_split = data_split.lower()
        self.task_construction_mode = str(self.config.task_construction_mode)

        # Set subjects based on mode
        if train_subjects is None:
            raise ValueError("Must provide train_subjects")

        self.train_subjects = [int(subject) for subject in train_subjects]
        self.test_subject = None if test_subject is None else int(test_subject)
        if test_subjects is not None:
            self.test_subjects = [int(subject) for subject in test_subjects]
        elif self.test_subject is not None:
            self.test_subjects = [self.test_subject]
        else:
            self.test_subjects = []

        if mode == "train":
            self.active_subjects = self.train_subjects
            self.tasks_per_epoch = self.config.tasks_per_epoch
        elif mode == "val":
            # Use a subset of training subjects for validation
            self.active_subjects = self.train_subjects[-5:]  # Last 5 subjects
            self.tasks_per_epoch = self.config.val_tasks
        else:  # test
            self.active_subjects = self.test_subjects
            if not self.active_subjects:
                raise ValueError("Must provide test_subject or test_subjects for test mode")
            self.tasks_per_epoch = self.config.subject_eval_tasks

        self.split_normalization_stats = None
        if self.config.task_normalize_mode == "split":
            self.split_normalization_stats = (
                self.dataset.compute_split_normalization_stats(
                    self.active_subjects,
                    split=self.data_split,
                )
            )

        self.n_way = self.config.n_way

        # Precompute shapes
        self.support_size = self.n_way * self.k_shot
        self.query_size = self.n_way * self.q_query
        self.seq_len = self.config.sequence_length
        self.n_sensors = self.config.num_sensors

    def __len__(self) -> int:
        """Number of tasks per epoch."""
        return self.tasks_per_epoch

    def __iter__(self) -> Generator[Dict[str, np.ndarray], None, None]:
        """Iterate over tasks."""
        for _ in range(self.tasks_per_epoch):
            yield self._sample_task()

    def _sample_task(self) -> Dict[str, np.ndarray]:
        """Sample a single task."""
        normalize_mode = self.config.task_normalize_mode
        use_base_index = self.mode == "test"
        active_subjects = [int(subject) for subject in self.active_subjects]
        if not active_subjects:
            raise ValueError(f"No active subjects configured for mode={self.mode}")

        mode = self.task_construction_mode
        if mode == "single_subject":
            sampled_subject = int(self.rng.choice(active_subjects))
            return self.dataset.sample_task(
                subject=sampled_subject,
                k_shot=self.k_shot,
                q_query=self.q_query,
                normalize_mode=normalize_mode,
                rng=self.rng,
                allow_partial_query=use_base_index,
                split_normalization_stats=self.split_normalization_stats,
                split=self.data_split,
                use_base_index=use_base_index,
            )

        if mode == "cross_subject":
            if len(active_subjects) < 2:
                sampled_subject = active_subjects[0]
                return self.dataset.sample_task(
                    subject=sampled_subject,
                    k_shot=self.k_shot,
                    q_query=self.q_query,
                    normalize_mode=normalize_mode,
                    rng=self.rng,
                    allow_partial_query=use_base_index,
                    split_normalization_stats=self.split_normalization_stats,
                    split=self.data_split,
                    use_base_index=use_base_index,
                )
            support_subject, query_subject = self.rng.choice(
                active_subjects,
                size=2,
                replace=False,
            ).tolist()
            return self.dataset.sample_task_cross_subjects(
                support_subject=int(support_subject),
                query_subject=int(query_subject),
                k_shot=self.k_shot,
                q_query=self.q_query,
                normalize_mode=normalize_mode,
                rng=self.rng,
                split_normalization_stats=self.split_normalization_stats,
                split=self.data_split,
                use_base_index=use_base_index,
            )

        if mode == "mixed":
            if len(active_subjects) < 2:
                sampled_subject = active_subjects[0]
                return self.dataset.sample_task(
                    subject=sampled_subject,
                    k_shot=self.k_shot,
                    q_query=self.q_query,
                    normalize_mode=normalize_mode,
                    rng=self.rng,
                    allow_partial_query=use_base_index,
                    split_normalization_stats=self.split_normalization_stats,
                    split=self.data_split,
                    use_base_index=use_base_index,
                )
            permuted_subjects = self.rng.permutation(active_subjects).tolist()
            n_support = max(1, len(permuted_subjects) // 2)
            support_subjects = permuted_subjects[:n_support]
            query_subjects = permuted_subjects[n_support:]
            if not query_subjects:
                query_subjects = support_subjects[-1:]
                support_subjects = support_subjects[:-1]
            if not support_subjects:
                support_subjects = query_subjects[:1]
                query_subjects = query_subjects[1:]
            if not query_subjects:
                # Degenerate case fallback.
                return self.dataset.sample_task(
                    subject=int(support_subjects[0]),
                    k_shot=self.k_shot,
                    q_query=self.q_query,
                    normalize_mode=normalize_mode,
                    rng=self.rng,
                    allow_partial_query=use_base_index,
                    split_normalization_stats=self.split_normalization_stats,
                    split=self.data_split,
                    use_base_index=use_base_index,
                )
            return self.dataset.sample_task_mixed_subject_pools(
                support_subjects=[int(subject) for subject in support_subjects],
                query_subjects=[int(subject) for subject in query_subjects],
                k_shot=self.k_shot,
                q_query=self.q_query,
                normalize_mode=normalize_mode,
                rng=self.rng,
                split_normalization_stats=self.split_normalization_stats,
                split=self.data_split,
                use_base_index=use_base_index,
            )

        raise ValueError(
            f"Unknown task_construction_mode='{mode}'. "
            "Use one of: single_subject, cross_subject, mixed."
        )

    def get_task(self, subject: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get a single task, optionally from a specific subject.

        Args:
            subject: Optional subject ID (random if None)

        Returns:
            Task dictionary
        """
        normalize_mode = self.config.task_normalize_mode
        if subject is not None:
            return self.dataset.sample_task(
                subject,
                self.k_shot,
                self.q_query,
                normalize_mode=normalize_mode,
                rng=self.rng,
                allow_partial_query=self.mode == "test",
                split_normalization_stats=self.split_normalization_stats,
                split=self.data_split,
                use_base_index=self.mode == "test",
            )
        return self._sample_task()

    def get_test_task(self, k_shot: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get an task from the test subject.

        Args:
            k_shot: Override default k_shot

        Returns:
            Task dictionary
        """
        if len(self.active_subjects) == 1:
            return self.dataset.sample_task(
                subject=self.active_subjects[0],
                k_shot=k_shot or self.k_shot,
                q_query=self.q_query,
                normalize_mode=self.config.task_normalize_mode,
                rng=self.rng,
                allow_partial_query=True,
                split_normalization_stats=self.split_normalization_stats,
                split=self.data_split,
                use_base_index=True,
            )
        return self.dataset.sample_task_from_subjects(
            subjects=self.active_subjects,
            k_shot=k_shot or self.k_shot,
            q_query=self.q_query,
            normalize_mode=self.config.task_normalize_mode,
            rng=self.rng,
            allow_partial_query=True,
            split_normalization_stats=self.split_normalization_stats,
            split=self.data_split,
            use_base_index=True,
        )

    def as_tf_dataset(self, batch_size: int = 1, prefetch: int = 2) -> tf.data.Dataset:
        """
        Convert to TensorFlow Dataset.

        Args:
            batch_size: Batch size (number of tasks)
            prefetch: Prefetch buffer size

        Returns:
            tf.data.Dataset yielding batched tasks
        """
        # Define output signature
        output_signature = {
            "support_X": tf.TensorSpec(
                shape=(self.support_size, self.seq_len, self.n_sensors),
                dtype=tf.float32,
            ),
            "support_y": tf.TensorSpec(shape=(self.support_size,), dtype=tf.int32),
            "query_X": tf.TensorSpec(
                shape=(None, self.seq_len, self.n_sensors), dtype=tf.float32
            ),
            "query_y": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "subject": tf.TensorSpec(shape=(), dtype=tf.int32),
        }

        def generator():
            for task in self:
                yield {
                    "support_X": task["support_X"].astype(np.float32),
                    "support_y": task["support_y"].astype(np.int32),
                    "query_X": task["query_X"].astype(np.float32),
                    "query_y": task["query_y"].astype(np.int32),
                    "subject": np.int32(task["subject"]),
                }

        dataset = tf.data.Dataset.from_generator(
            generator, output_signature=output_signature
        )

        if batch_size > 1:
            dataset = dataset.batch(batch_size)

        return dataset.prefetch(prefetch)

    def as_multimodal_tf_dataset(
        self, batch_size: int = 1, prefetch: int = 2
    ) -> tf.data.Dataset:
        """
        Convert to TensorFlow Dataset with separate modality tensors.

        This is useful when the model has separate encoders per modality.

        Returns:
            tf.data.Dataset where each sample has modality-specific tensors
        """
        modality_names = self.config.modality_names

        def generator():
            for task in self:
                # Split by modality
                support_modalities = {
                    name: task["support_X"][:, :, i : i + 1].astype(np.float32)
                    for i, name in enumerate(modality_names)
                }
                query_modalities = {
                    name: task["query_X"][:, :, i : i + 1].astype(np.float32)
                    for i, name in enumerate(modality_names)
                }

                yield {
                    "support": support_modalities,
                    "support_y": task["support_y"].astype(np.int32),
                    "query": query_modalities,
                    "query_y": task["query_y"].astype(np.int32),
                    "subject": np.int32(task["subject"]),
                }

        # Build output signature
        modality_spec = {
            name: tf.TensorSpec(shape=(None, self.seq_len, 1), dtype=tf.float32)
            for name in modality_names
        }

        output_signature = {
            "support": modality_spec,
            "support_y": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "query": modality_spec.copy(),
            "query_y": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "subject": tf.TensorSpec(shape=(), dtype=tf.int32),
        }

        dataset = tf.data.Dataset.from_generator(
            generator, output_signature=output_signature
        )

        if batch_size > 1:
            dataset = dataset.batch(batch_size)

        return dataset.prefetch(prefetch)
