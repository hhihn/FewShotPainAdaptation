import numpy as np
import tensorflow as tf
from typing import List, Dict, Optional, Generator

import logging

from utils.logger import setup_logger

from data_loaders.pain_meta_dataset import PainMetaDataset


class SixWayKShotSampler:
    """
    6-Way-K-Shot episodic sampler for meta-learning.

    This sampler generates episodes for training and evaluating meta-learning
    models on the pain dataset. Each episode contains:
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
        episodes_per_epoch: int = 100,
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
            episodes_per_epoch: Number of episodes per epoch
            seed: Random seed
        """
        self.logger = setup_logger(__name__, level=logging.DEBUG)
        self.dataset = dataset
        self.mode = mode
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch
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

        if seed is not None:
            np.random.seed(seed)

    def __len__(self) -> int:
        """Number of episodes per epoch."""
        return self.episodes_per_epoch

    def __iter__(self) -> Generator[Dict[str, np.ndarray], None, None]:
        """Iterate over episodes."""
        for _ in range(self.episodes_per_epoch):
            yield self._sample_episode()

    def _sample_episode(self) -> Dict[str, np.ndarray]:
        """Sample a single episode."""

        episode_dict = {"support_X": [], "support_y": [], "query_X": [], "query_y": []}

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
                episode_dict["support_X"].append(x)
                episode_dict["support_y"].append(class_id)

            # Load query samples
            for idx in query_indices:
                x = self.dataset.X[idx]
                episode_dict["query_X"].append(x)
                episode_dict["query_y"].append(class_id)

        # Stack into arrays
        episode_dict["support_X"] = np.stack(episode_dict["support_X"], axis=0)
        episode_dict["support_y"] = np.array(episode_dict["support_y"])
        episode_dict["query_X"] = np.stack(episode_dict["query_X"], axis=0)
        episode_dict["query_y"] = np.array(episode_dict["query_y"])

        return episode_dict

    def get_episode(self, subject: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get a single episode, optionally from a specific subject.

        Args:
            subject: Optional subject ID (random if None)

        Returns:
            Episode dictionary
        """
        if subject is None:
            subject = np.random.choice(self.active_subjects)
        normalize_mode = "subject" if self.mode == "train" else "support"
        return self.dataset.sample_episode(
            subject, self.k_shot, self.q_query, normalize_mode=normalize_mode
        )

    def get_test_episode(self, k_shot: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get an episode from the test subject.

        Args:
            k_shot: Override default k_shot

        Returns:
            Episode dictionary
        """
        return self.dataset.sample_episode(
            subject=self.test_subject,
            k_shot=k_shot or self.k_shot,
            q_query=self.q_query,
            normalize_mode="support",
        )

    def as_tf_dataset(self, batch_size: int = 1, prefetch: int = 2) -> tf.data.Dataset:
        """
        Convert to TensorFlow Dataset.

        Args:
            batch_size: Batch size (number of episodes)
            prefetch: Prefetch buffer size

        Returns:
            tf.data.Dataset yielding batched episodes
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
            for episode in self:
                yield {
                    "support_X": episode["support_X"].astype(np.float32),
                    "support_y": episode["support_y"].astype(np.int32),
                    "query_X": episode["query_X"].astype(np.float32),
                    "query_y": episode["query_y"].astype(np.int32),
                    "subject": np.int32(episode["subject"]),
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
            for episode in self:
                # Split by modality
                support_modalities = {
                    name: episode["support_X"][:, :, i : i + 1].astype(np.float32)
                    for i, name in enumerate(modality_names)
                }
                query_modalities = {
                    name: episode["query_X"][:, :, i : i + 1].astype(np.float32)
                    for i, name in enumerate(modality_names)
                }

                yield {
                    "support": support_modalities,
                    "support_y": episode["support_y"].astype(np.int32),
                    "query": query_modalities,
                    "query_y": episode["query_y"].astype(np.int32),
                    "subject": np.int32(episode["subject"]),
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
