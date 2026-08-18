"""BioVid loading and non-episodic mini-batch construction for PainNAS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from data_loaders.pain_ds_config import PainDatasetConfig
from data_loaders.pain_meta_dataset import PainMetaDataset
from painnas.config import PainNASConfig


@dataclass
class BioVidArrays:
    """In-memory BioVid arrays with class labels remapped to consecutive indices."""

    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    split_codes: np.ndarray
    subject_keys: dict[int, str]
    train_split_code: int = 0
    test_split_code: int = 1

    @property
    def unique_subjects(self) -> np.ndarray:
        return np.unique(self.subjects)

    @property
    def sequence_length(self) -> int:
        return int(self.X.shape[1])

    @property
    def num_modalities(self) -> int:
        return int(self.X.shape[2])

    def validate(self, config: PainNASConfig, *, require_expected_subjects: bool) -> None:
        if self.X.ndim != 3:
            raise ValueError(f"Expected X shaped [samples,time,modalities], got {self.X.shape}")
        if self.X.shape[0] != len(self.y) or len(self.y) != len(self.subjects):
            raise ValueError("X, y, and subjects must contain the same number of samples")
        if len(self.split_codes) != len(self.y):
            raise ValueError("split_codes must contain one value per sample")
        if self.num_modalities != len(config.modalities):
            raise ValueError(
                f"Expected {len(config.modalities)} modalities, got {self.num_modalities}"
            )
        if require_expected_subjects and len(self.unique_subjects) != config.expected_subjects:
            raise ValueError(
                f"Expected {config.expected_subjects} subjects, got {len(self.unique_subjects)}"
            )
        if require_expected_subjects and self.sequence_length != config.expected_sequence_length:
            raise ValueError(
                f"Expected sequence length {config.expected_sequence_length}, "
                f"got {self.sequence_length}"
            )
        expected_labels = set(range(config.num_classes))
        if set(np.unique(self.y).tolist()) != expected_labels:
            raise ValueError(
                "PainNAS labels must be remapped to every consecutive class index "
                f"in {sorted(expected_labels)}"
            )
        if not np.all(np.isfinite(self.X)):
            raise ValueError("Input arrays contain NaN or infinite values")


@dataclass(frozen=True)
class FoldIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    source_subjects: tuple[int, ...]
    target_subject: int


@dataclass(frozen=True)
class NestedFoldIndices:
    inner_train: np.ndarray
    inner_validation: np.ndarray
    final_train: np.ndarray
    test: np.ndarray
    source_subjects: tuple[int, ...]
    inner_train_subjects: tuple[int, ...]
    inner_validation_subjects: tuple[int, ...]
    target_subject: int


@dataclass(frozen=True)
class CrossFittedSubjectPlan:
    """Deterministic outer blocks and their inner development folds."""

    outer_blocks: tuple[tuple[int, ...], ...]
    inner_folds_by_block: tuple[tuple[tuple[int, ...], ...], ...]
    seed: int

    def validate(self, subjects: Iterable[int]) -> None:
        known = {int(value) for value in subjects}
        flattened_outer = [subject for block in self.outer_blocks for subject in block]
        if len(flattened_outer) != len(set(flattened_outer)):
            raise RuntimeError("Outer subject blocks overlap")
        if set(flattened_outer) != known:
            raise RuntimeError("Outer subject blocks do not cover all subjects")
        for block, inner_folds in zip(
            self.outer_blocks, self.inner_folds_by_block
        ):
            development = known - set(block)
            flattened_inner = [
                subject for fold in inner_folds for subject in fold
            ]
            if len(flattened_inner) != len(set(flattened_inner)):
                raise RuntimeError("Inner subject folds overlap")
            if set(flattened_inner) != development:
                raise RuntimeError(
                    "Inner subject folds do not cover the development subjects"
                )


def _balanced_subject_folds(
    subjects: Iterable[int], *, fold_count: int, seed: int
) -> tuple[tuple[int, ...], ...]:
    ordered = np.asarray(sorted(int(value) for value in subjects), dtype=np.int32)
    if fold_count < 2 or fold_count > len(ordered):
        raise ValueError("fold_count must be between 2 and the subject count")
    shuffled = np.random.default_rng(seed).permutation(ordered)
    return tuple(
        tuple(sorted(int(value) for value in fold.tolist()))
        for fold in np.array_split(shuffled, fold_count)
    )


def build_cross_fitted_subject_plan(
    subjects: Iterable[int], *, outer_block_count: int, inner_fold_count: int,
    seed: int,
) -> CrossFittedSubjectPlan:
    """Build balanced outer blocks and inner folds without subject overlap."""

    ordered = tuple(sorted(int(value) for value in subjects))
    outer_blocks = _balanced_subject_folds(
        ordered, fold_count=outer_block_count, seed=seed
    )
    known = set(ordered)
    inner_folds_by_block = tuple(
        _balanced_subject_folds(
            known - set(block),
            fold_count=inner_fold_count,
            seed=seed + (block_index + 1) * 100_000,
        )
        for block_index, block in enumerate(outer_blocks)
    )
    plan = CrossFittedSubjectPlan(
        outer_blocks=outer_blocks,
        inner_folds_by_block=inner_folds_by_block,
        seed=int(seed),
    )
    plan.validate(ordered)
    return plan


def load_biovid_binary(data_dir: str, config: PainNASConfig) -> BioVidArrays:
    """Load selected BioVid classes and remap them to consecutive indices.

    The historical function name is retained for API compatibility.  The selected
    raw class indices are controlled by ``config.raw_class_ids`` and may describe
    either a binary or a multi-class task.
    """

    dataset_config = PainDatasetConfig(
        dataset_source="biovid_part_a",
        encoder_backend="eegnet",
        task_class_ids=config.raw_class_ids,
        biovid_modalities=config.modalities,
        modalities=None,
        enable_window_shift_augmentation=False,
        seed=config.seed,
    )
    dataset = PainMetaDataset(
        data_dir=data_dir,
        config=dataset_config,
        normalize=False,
        normalize_per_subject=False,
    )
    class_mask = np.isin(dataset.y, config.raw_class_ids)
    raw_y = dataset.y[class_mask]
    raw_to_index = {raw_class: index for index, raw_class in enumerate(config.raw_class_ids)}
    remapped_y = np.asarray([raw_to_index[int(label)] for label in raw_y], dtype=np.int32)
    subject_keys = {
        int(key): str(value)
        for key, value in dataset.predefined_subject_int_to_key.items()
    }
    arrays = BioVidArrays(
        X=np.ascontiguousarray(dataset.X[class_mask], dtype=np.float32),
        y=remapped_y,
        subjects=np.ascontiguousarray(dataset.subjects[class_mask], dtype=np.int32),
        split_codes=np.ascontiguousarray(
            dataset.sample_split_codes[class_mask], dtype=np.int8
        ),
        subject_keys=subject_keys,
        train_split_code=dataset.split_name_to_code["train"],
        test_split_code=dataset.split_name_to_code["test"],
    )
    arrays.validate(config, require_expected_subjects=True)
    return arrays


def deterministic_search_subject_split(
    subjects: Iterable[int], *, validation_subjects: int, seed: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ordered = np.asarray(sorted(int(subject) for subject in subjects), dtype=np.int32)
    if validation_subjects <= 0 or validation_subjects >= len(ordered):
        raise ValueError("validation_subjects must be between 1 and subject_count - 1")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ordered)
    validation = tuple(sorted(int(value) for value in shuffled[:validation_subjects]))
    training = tuple(sorted(int(value) for value in shuffled[validation_subjects:]))
    return training, validation


def indices_for_subjects(
    arrays: BioVidArrays, subjects: Iterable[int], *, split_code: int
) -> np.ndarray:
    selected = np.asarray(tuple(int(subject) for subject in subjects), dtype=np.int32)
    mask = np.isin(arrays.subjects, selected) & (arrays.split_codes == split_code)
    return np.flatnonzero(mask).astype(np.int64, copy=False)


def build_loso_fold_indices(arrays: BioVidArrays, target_subject: int) -> FoldIndices:
    target_subject = int(target_subject)
    all_subjects = tuple(int(value) for value in arrays.unique_subjects)
    if target_subject not in all_subjects:
        raise ValueError(f"Unknown target subject: {target_subject}")
    source_subjects = tuple(value for value in all_subjects if value != target_subject)
    train = indices_for_subjects(
        arrays, source_subjects, split_code=arrays.train_split_code
    )
    validation = indices_for_subjects(
        arrays, source_subjects, split_code=arrays.test_split_code
    )
    test = indices_for_subjects(
        arrays, (target_subject,), split_code=arrays.test_split_code
    )
    if not len(train) or not len(validation) or not len(test):
        raise ValueError(f"Fold {target_subject} contains an empty split")
    for split_name, split_indices in (
        ("train", train),
        ("validation", validation),
    ):
        if np.any(arrays.subjects[split_indices] == target_subject):
            raise RuntimeError(f"Target subject leaked into {split_name}")
    return FoldIndices(
        train=train,
        validation=validation,
        test=test,
        source_subjects=source_subjects,
        target_subject=target_subject,
    )


def build_nested_loso_fold_indices(
    arrays: BioVidArrays,
    target_subject: int,
    *,
    validation_subjects: int,
    seed: int,
) -> NestedFoldIndices:
    """Build a subject-disjoint inner NAS split for one outer LOSO fold."""

    target_subject = int(target_subject)
    all_subjects = tuple(int(value) for value in arrays.unique_subjects)
    if target_subject not in all_subjects:
        raise ValueError(f"Unknown target subject: {target_subject}")
    source_subjects = tuple(
        subject for subject in all_subjects if subject != target_subject
    )
    inner_train_subjects, inner_validation_subjects = (
        deterministic_search_subject_split(
            source_subjects,
            validation_subjects=validation_subjects,
            seed=seed,
        )
    )
    inner_train = indices_for_subjects(
        arrays, inner_train_subjects, split_code=arrays.train_split_code
    )
    inner_validation = indices_for_subjects(
        arrays, inner_validation_subjects, split_code=arrays.test_split_code
    )
    final_train = indices_for_subjects(
        arrays, source_subjects, split_code=arrays.train_split_code
    )
    test = indices_for_subjects(
        arrays, (target_subject,), split_code=arrays.test_split_code
    )
    split_indices = {
        "inner_train": inner_train,
        "inner_validation": inner_validation,
        "final_train": final_train,
        "test": test,
    }
    empty = [name for name, indices in split_indices.items() if not len(indices)]
    if empty:
        raise ValueError(
            f"Nested fold {target_subject} contains empty splits: {', '.join(empty)}"
        )
    for split_name in ("inner_train", "inner_validation", "final_train"):
        split_values = split_indices[split_name]
        if np.any(arrays.subjects[split_values] == target_subject):
            raise RuntimeError(f"Target subject leaked into {split_name}")
    if set(inner_train_subjects) & set(inner_validation_subjects):
        raise RuntimeError("Inner train and validation subjects overlap")
    if set(inner_train_subjects) | set(inner_validation_subjects) != set(
        source_subjects
    ):
        raise RuntimeError("Inner split does not cover every source subject")
    return NestedFoldIndices(
        inner_train=inner_train,
        inner_validation=inner_validation,
        final_train=final_train,
        test=test,
        source_subjects=source_subjects,
        inner_train_subjects=inner_train_subjects,
        inner_validation_subjects=inner_validation_subjects,
        target_subject=target_subject,
    )


def compute_source_normalization(
    X: np.ndarray, indices: np.ndarray, *, chunk_size: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-modality z-score statistics without copying the full fold."""

    if len(indices) == 0:
        raise ValueError("Cannot normalize an empty source split")
    sums = np.zeros(X.shape[2], dtype=np.float64)
    sums_of_squares = np.zeros(X.shape[2], dtype=np.float64)
    count = 0
    for start in range(0, len(indices), chunk_size):
        chunk = X[indices[start : start + chunk_size]].astype(np.float64, copy=False)
        sums += np.sum(chunk, axis=(0, 1))
        sums_of_squares += np.sum(np.square(chunk), axis=(0, 1))
        count += int(chunk.shape[0] * chunk.shape[1])
    mean = sums / count
    variance = np.maximum(sums_of_squares / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std = np.maximum(std, 1e-8)
    return mean.astype(np.float32), std.astype(np.float32)


def make_tf_dataset(
    arrays: BioVidArrays,
    indices: np.ndarray,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    training: bool,
    seed: int,
):
    """Build a vectorized channels-last 2D CNN input pipeline."""

    import tensorflow as tf

    selected_X = arrays.X[indices]
    selected_y = arrays.y[indices]
    class_count = int(np.max(arrays.y)) + 1
    dataset = tf.data.Dataset.from_tensor_slices((selected_X, selected_y))
    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(indices), 20_000),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    mean_tensor = tf.constant(mean.reshape(1, 1, -1), dtype=tf.float32)
    std_tensor = tf.constant(std.reshape(1, 1, -1), dtype=tf.float32)

    def prepare_batch(batch_X, batch_y):
        normalized = (tf.cast(batch_X, tf.float32) - mean_tensor) / std_tensor
        transposed = tf.transpose(normalized, perm=(0, 2, 1))
        model_input = tf.expand_dims(transposed, axis=-1)
        one_hot_y = tf.one_hot(batch_y, depth=class_count, dtype=tf.float32)
        return model_input, one_hot_y

    dataset = dataset.map(
        prepare_batch,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)
