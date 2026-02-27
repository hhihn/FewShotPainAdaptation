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
        k_shot: int = 5,
        q_query: int = 5,
        tasks_per_epoch: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Initialize the sampler.

        Args:
            dataset: PainMetaDataset instance
            mode: 'train', 'val', or 'test'
            train_subjects: List of training subject IDs
            test_subject: Held-out test subject ID
            k_shot: Number of support samples per class
            q_query: Number of query samples per class
            tasks_per_epoch: Number of tasks per epoch
            seed: Random seed
        """
        self.logger = setup_logger(__name__)
        self.dataset = dataset
        self.mode = mode
        self.k_shot = k_shot
        self.q_query = q_query
        self.tasks_per_epoch = tasks_per_epoch
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        # Set subjects based on mode
        if train_subjects is None or test_subject is None:
            raise ValueError("Must provide train_subjects and test_subject")

        self.train_subjects = train_subjects
        self.test_subject = test_subject

        if mode == "train":
            self.active_subjects = train_subjects
        elif mode == "val":
            # Use a subset of training subjects for validation
            self.active_subjects = train_subjects[-5:]  # Last 5 subjects
        else:  # test
            self.active_subjects = [test_subject]

        self.config = dataset.config
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

        task_dict = {"support_X": [], "support_y": [], "query_X": [], "query_y": []}

        # For each of the 6 pain levels
        for class_id in range(self.config.n_way):
            # Pool all samples from all training subjects for this class
            pooled_indices = []

            for subject in self.active_subjects:
                if (
                    subject in self.dataset.index
                    and class_id in self.dataset.index[subject]
                ):
                    pooled_indices.extend(self.dataset.index[subject][class_id])

            if len(pooled_indices) < self.config.k_shot + self.config.q_query:
                self.logger.warning(
                    f"Class {class_id} has only {len(pooled_indices)} samples across "
                    f"all {len(self.active_subjects)} training subjects"
                )

            # Randomly sample K + Q samples from the pooled indices
            sampled_indices = self.rng.choice(
                pooled_indices,
                size=min(self.config.k_shot + self.config.q_query, len(pooled_indices)),
                replace=False,
            )

            support_indices = sampled_indices[: self.config.k_shot]
            query_indices = sampled_indices[self.config.k_shot :]

            # Load support samples
            for idx in support_indices:
                x = self.dataset.X[idx]
                task_dict["support_X"].append(x)
                task_dict["support_y"].append(class_id)

            # Load query samples
            for idx in query_indices:
                x = self.dataset.X[idx]
                task_dict["query_X"].append(x)
                task_dict["query_y"].append(class_id)

        # Stack into arrays
        task_dict["support_X"] = np.stack(task_dict["support_X"], axis=0)
        task_dict["support_y"] = np.array(task_dict["support_y"])
        task_dict["query_X"] = np.stack(task_dict["query_X"], axis=0)
        task_dict["query_y"] = np.array(task_dict["query_y"])

        return task_dict

    def get_task(self, subject: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get a single task, optionally from a specific subject.

        Args:
            subject: Optional subject ID (random if None)

        Returns:
            Task dictionary
        """
        if subject is None:
            subject = self.rng.choice(self.active_subjects)
        normalize_mode = "subject" if self.mode == "train" else "support"
        return self.dataset.sample_task(
            subject,
            self.k_shot,
            self.q_query,
            normalize_mode=normalize_mode,
            rng=self.rng,
        )

    def get_test_task(self, k_shot: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get an task from the test subject.

        Args:
            k_shot: Override default k_shot

        Returns:
            Task dictionary
        """
        return self.dataset.sample_task(
            subject=self.test_subject,
            k_shot=k_shot or self.k_shot,
            q_query=self.q_query,
            normalize_mode="support",
            rng=self.rng,
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
                shape=(self.query_size, self.seq_len, self.n_sensors), dtype=tf.float32
            ),
            "query_y": tf.TensorSpec(shape=(self.query_size,), dtype=tf.int32),
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
