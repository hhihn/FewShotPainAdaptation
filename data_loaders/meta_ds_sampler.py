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

    VALIDATION_FALLBACK_K_SHOT = 4
    VALIDATION_FALLBACK_Q_QUERY = 4

    def __init__(
        self,
        dataset: PainMetaDataset,
        mode: str = "train",
        train_subjects: Optional[List[int]] = None,
        test_subject: Optional[int] = None,
        test_subjects: Optional[List[int]] = None,
        data_split: str = "all",
        seed: Optional[int] = None,
        normalization_stats: Optional[Dict[str, np.ndarray]] = None,
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
        self._reported_task_size_signatures = set()

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
            # Validation subjects are selected by the cross-validator.
            self.active_subjects = self.train_subjects
            self.tasks_per_epoch = self.config.val_tasks
        else:  # test
            self.active_subjects = self.test_subjects
            if not self.active_subjects:
                raise ValueError(
                    "Must provide test_subject or test_subjects for test mode"
                )
            self.tasks_per_epoch = self.config.heldout_eval_tasks
        self.active_subjects_array = np.asarray(self.active_subjects, dtype=np.int32)
        if self.mode == "train" and self.task_construction_mode == "single_subject":
            self._drop_undersized_train_subjects()
        if self.mode in {"val", "test"}:
            self._apply_eval_task_size_fallback()

        self.logger.info(
            "Initialized sampler: "
            f"mode={self.mode}, split={self.data_split}, "
            f"loso_subject={self.test_subject}, "
            f"num_active_subjects={len(self.active_subjects)}"
        )

        self.split_normalization_stats = None
        if self.config.task_normalize_mode == "split":
            if normalization_stats is not None:
                self.split_normalization_stats = normalization_stats
            else:
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
        """Return the number of tasks yielded per epoch.

        The value is derived from the sampler mode and dataset configuration.
        """
        return self.tasks_per_epoch

    def __iter__(self) -> Generator[Dict[str, np.ndarray], None, None]:
        """Iterate over sampled episodic tasks.

        Yields:
            Task dictionaries containing support/query arrays and labels.
        """
        for _ in range(self.tasks_per_epoch):
            yield self._sample_task()

    def _min_available_samples_per_class(
        self,
        subjects: List[int],
        *,
        use_base_index: bool,
        pool_subjects: bool = True,
    ) -> int:
        """Return the smallest per-class sample pool for selected subjects.

        Args:
            subjects: Subject IDs to inspect.
            use_base_index: Whether to ignore window-shift expanded references.
            pool_subjects: Whether samples are pooled across subjects per class.
        """
        split_index = self.dataset._get_sampling_index_for_split(
            self.data_split,
            use_base_index=use_base_index,
        )
        selected_subjects = [int(subject) for subject in subjects]
        if not pool_subjects:
            return min(
                int(len(split_index[subject][class_id]))
                for subject in selected_subjects
                for class_id in range(self.config.n_way)
            )
        return min(
            int(
                sum(
                    len(split_index[subject][class_id]) for subject in selected_subjects
                )
            )
            for class_id in range(self.config.n_way)
        )

    def resolve_effective_task_size(
        self,
        k_shot: int | None = None,
        q_query: int | None = None,
    ) -> tuple[int, int]:
        """Return task size after applying evaluation-only fallback rules.

        Training samplers keep the requested task size. Validation and test
        samplers use fixed fallback sizes when the requested support+query total
        exceeds the raw evaluation pool.
        """
        requested_k = int(self.k_shot if k_shot is None else k_shot)
        requested_q = int(self.q_query if q_query is None else q_query)
        if self.mode not in {"val", "test"}:
            return requested_k, requested_q

        fallback_total = int(self.VALIDATION_FALLBACK_K_SHOT) + int(
            self.VALIDATION_FALLBACK_Q_QUERY
        )
        requested_total = requested_k + requested_q
        if requested_total > fallback_total:
            return self.VALIDATION_FALLBACK_K_SHOT, self.VALIDATION_FALLBACK_Q_QUERY

        available_count = self._min_available_samples_per_class(
            self.active_subjects,
            use_base_index=True,
            pool_subjects=self.task_construction_mode != "single_subject",
        )
        if requested_total <= available_count:
            return requested_k, requested_q
        # PainMonit: a subject with a short class (subject 15 has 7 in class 5)
        # cannot supply the configured 4+4=8. Shrink k=q to fit the available
        # samples instead of the fixed fallback, which would still be too large.
        fitted = max(1, available_count // 2)
        return fitted, fitted

    def _drop_undersized_train_subjects(self) -> None:
        """Drop train subjects that cannot supply a full k+q task in every class.

        Single-subject tasks take support and query from one subject, so a
        subject whose smallest class pool is below ``k_shot + q_query`` cannot
        form a task at all. Window-shift augmentation normally hides this by
        expanding each trial into many windows; with augmentation disabled the
        raw trial counts apply, and PainMonit subject 15 has only 7 samples in
        raw class 5 against a required 8.

        Dropping the subject keeps every task at the configured size, so a
        windowing-off run stays comparable to a windowed one. Shrinking k/q
        instead - the fallback used for val/test - would change the task size
        for all subjects and confound that comparison.
        """
        required = int(self.k_shot) + int(self.q_query)
        split_index = self.dataset._get_sampling_index_for_split(
            self.data_split,
            use_base_index=False,
        )
        usable: List[int] = []
        dropped: List[tuple[int, int]] = []
        for subject in self.active_subjects:
            smallest = min(
                int(len(split_index[subject][class_id]))
                for class_id in range(self.config.n_way)
            )
            if smallest >= required:
                usable.append(subject)
            else:
                dropped.append((subject, smallest))

        if not dropped:
            return
        if not usable:
            raise ValueError(
                f"No train subject can supply k_shot+q_query={required} samples "
                f"in every class for split={self.data_split}."
            )

        self.active_subjects = usable
        self.active_subjects_array = np.asarray(self.active_subjects, dtype=np.int32)
        self.logger.warning(
            "Dropped %d of %d train subject(s) with fewer than %d samples in "
            "some class: %s. Remaining train subjects: %d.",
            len(dropped),
            len(dropped) + len(usable),
            required,
            ", ".join(f"subject {s} (min {n})" for s, n in dropped),
            len(usable),
        )

    def _apply_eval_task_size_fallback(self) -> None:
        """Use fixed evaluation task sizes when configured sizes are too large.

        Validation and test tasks fall back to stable k/q defaults when raw
        per-class subject pools cannot satisfy the configured request.
        """
        effective_k, effective_q = self.resolve_effective_task_size(
            self.k_shot,
            self.q_query,
        )
        if (effective_k, effective_q) == (int(self.k_shot), int(self.q_query)):
            return

        original_k = int(self.k_shot)
        original_q = int(self.q_query)
        available_count = self._min_available_samples_per_class(
            self.active_subjects,
            use_base_index=True,
            pool_subjects=self.task_construction_mode != "single_subject",
        )
        self.k_shot = int(effective_k)
        self.q_query = int(effective_q)
        self.logger.info(
            "%s task size k=%s, q=%s exceeds %s raw samples per class; "
            "using k=%s, q=%s for %s.",
            self.mode.capitalize(),
            original_k,
            original_q,
            available_count,
            self.k_shot,
            self.q_query,
            self.mode,
        )

    def _sample_task(self) -> Dict[str, np.ndarray]:
        """Sample one episodic task according to construction mode.

        The selected construction mode controls whether support and query data
        come from one subject, two subjects, or mixed subject pools.
        """
        normalize_mode = self.config.task_normalize_mode
        use_base_index = self.mode in {"val", "test"}
        active_subjects = self.active_subjects_array
        if active_subjects.size == 0:
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
            task = self.dataset.sample_task(
                subject,
                self.k_shot,
                self.q_query,
                normalize_mode=normalize_mode,
                rng=self.rng,
                allow_partial_query=self.mode in {"val", "test"},
                split_normalization_stats=self.split_normalization_stats,
                split=self.data_split,
                use_base_index=self.mode in {"val", "test"},
            )
            self._maybe_report_task_size_mismatch(task=task, requested_subject=subject)
            return task
        task = self._sample_task()
        self._maybe_report_task_size_mismatch(task=task, requested_subject=None)
        return task

    def _maybe_report_task_size_mismatch(
        self, task: Dict[str, np.ndarray], requested_subject: Optional[int]
    ) -> None:
        """Warn once when effective support/query sizes differ from requested sizes.

        Args:
            task: Sampled task dictionary.
            requested_subject: Explicit subject requested by the caller, if any.
        """
        support_y = np.asarray(task["support_y"], dtype=np.int32)
        query_y = np.asarray(task["query_y"], dtype=np.int32)

        support_counts = tuple(
            int(np.sum(support_y == class_id)) for class_id in range(self.n_way)
        )
        query_counts = tuple(
            int(np.sum(query_y == class_id)) for class_id in range(self.n_way)
        )
        expected_support = tuple(int(self.k_shot) for _ in range(self.n_way))
        expected_query = tuple(int(self.q_query) for _ in range(self.n_way))

        mismatch = support_counts != expected_support or query_counts != expected_query
        if not mismatch:
            return

        signature = (
            tuple(int(c) for c in support_counts),
            tuple(int(c) for c in query_counts),
            int(requested_subject) if requested_subject is not None else None,
        )
        if signature in self._reported_task_size_signatures:
            return
        self._reported_task_size_signatures.add(signature)

        self.logger.warning(
            "Episodic task size mismatch "
            f"(mode={self.mode}, split={self.data_split}, subject={requested_subject}): "
            f"requested_support_per_class={expected_support}, "
            f"requested_query_per_class={expected_query}, "
            f"actual_support_per_class={support_counts}, "
            f"actual_query_per_class={query_counts}, "
            f"support_total={len(support_y)}, query_total={len(query_y)}"
        )

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

    def as_tf_dataset(
        self, batch_size: int = 1, prefetch: int | None = None
    ) -> tf.data.Dataset:
        """
        Convert to TensorFlow Dataset.

        Args:
            batch_size: Batch size (number of tasks)
            prefetch: Prefetch buffer size; defaults to tf.data.AUTOTUNE

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
            """Yield task dictionaries matching the TensorFlow output signature.

            Values are cast to TensorFlow-compatible dtypes before yielding.
            """
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

        prefetch_buffer = tf.data.AUTOTUNE if prefetch is None else prefetch
        return dataset.prefetch(prefetch_buffer)

    def as_multimodal_tf_dataset(
        self, batch_size: int = 1, prefetch: int | None = None
    ) -> tf.data.Dataset:
        """
        Convert to TensorFlow Dataset with separate modality tensors.

        This is useful when the model has separate encoders per modality.

        Returns:
            tf.data.Dataset where each sample has modality-specific tensors
        """
        modality_names = self.config.modality_names

        def generator():
            """Yield modality-split task dictionaries for TensorFlow datasets.

            Support and query tensors are split into one channel per configured
            modality name.
            """
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

        prefetch_buffer = tf.data.AUTOTUNE if prefetch is None else prefetch
        return dataset.prefetch(prefetch_buffer)
