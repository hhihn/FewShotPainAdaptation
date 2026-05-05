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
            data_dir: Directory containing X/y/subject arrays as .npy or .npz files
            config: Dataset configuration
            normalize: Whether to normalize the data
            normalize_per_subject: If True, normalize per subject; else global normalization
        """
        self.logger = setup_logger("PainMetaDataset")
        self.config = config or PainDatasetConfig()
        self.task_class_ids = tuple(
            int(class_id) for class_id in self.config.task_class_ids
        )
        self.data_dir = Path(data_dir)
        self.logger.debug(f"Data directory: {self.data_dir}")
        self.normalize = normalize
        self.normalize_per_subject = normalize_per_subject
        self.window_shift_enabled = bool(self.config.enable_window_shift_augmentation)
        self.window_start_indices = np.array([], dtype=np.int64)
        self.window_length_samples = int(self.config.sequence_length)
        self.window_step_samples = 0
        self.window_start_min_idx = 0
        self.window_start_max_idx = 0
        self.has_predefined_split = bool(self.config.split_strategy == "predefined")
        self.sample_split_codes = np.array([], dtype=np.int8)
        self.split_code_to_name = {0: "train", 1: "test"}
        self.split_name_to_code = {v: k for k, v in self.split_code_to_name.items()}
        self.split_masks: Dict[str, np.ndarray] = {}
        self.index_by_split: Dict[str, Dict[int, Dict[int, np.ndarray]]] = {}
        self.base_index_by_split: Dict[str, Dict[int, Dict[int, np.ndarray]]] = {}

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
        if self.config.dataset_source == "biovid_part_a":
            self._load_biovid_part_a_data()
        else:
            self._load_painmonit_data()

        # Remove the trailing dimension if present (for CNN compatibility)
        if self.X.ndim == 4 and self.X.shape[-1] == 1:
            self.X = np.squeeze(self.X, axis=-1)

        self.X = self.X.astype(np.float32, copy=False)
        self.y = self.y.astype(np.int32, copy=False)
        self.subjects = self.subjects.astype(np.int32, copy=False)

        if self.sample_split_codes.size == 0:
            self.sample_split_codes = np.full(len(self.y), -1, dtype=np.int8)

        self.unique_subjects = np.unique(self.subjects)
        self.num_subjects = len(self.unique_subjects)
        self.logger.debug(f"Unique subjects: {self.unique_subjects}")
        self.logger.info(f"Number of subjects: {self.num_subjects}")
        self.logger.info(f"  Data shape: {self.X.shape}")
        self.logger.info(f"  Labels shape: {self.y.shape}")
        self.logger.info(f"  Number of subjects: {self.num_subjects}")
        self.logger.info(f"  Samples per subject: ~{len(self.y) // self.num_subjects}")
        self.logger.info(f"  Classes: {np.unique(self.y)}")
        self.logger.info(f"  Task classes: {self.task_class_ids}")

        if not self.split_masks:
            all_mask = np.ones(len(self.y), dtype=bool)
            self.split_masks = {"all": all_mask}
        else:
            self.split_masks["all"] = np.ones(len(self.y), dtype=bool)

        self.split_subjects = {
            split_name: sorted(
                np.unique(self.subjects[split_mask]).astype(int).tolist()
            )
            for split_name, split_mask in self.split_masks.items()
        }
        if self.has_predefined_split:
            train_count = int(
                np.sum(self.split_masks.get("train", np.zeros(0, dtype=bool)))
            )
            test_count = int(
                np.sum(self.split_masks.get("test", np.zeros(0, dtype=bool)))
            )
            self.logger.info(
                f"  Predefined split counts: train={train_count}, test={test_count}"
            )

    def _load_painmonit_data(self) -> None:
        """Load PainMonit-formatted arrays from a flat numpy directory."""
        if self.has_predefined_split:
            raise ValueError(
                "split_strategy='predefined' requires dataset_source='biovid_part_a'."
            )

        self.X = self._load_numpy_array(self.data_dir / self.config.data_path)[
            :, :, self.config.sensor_idx, :
        ]
        self.logger.info(f"X.shape: {self.X.shape}")
        self.y_onehot = self._load_numpy_array(self.data_dir / self.config.labels_path)
        self.logger.info(f"y_onehot.shape: {self.y_onehot.shape}")
        self.subjects = self._load_numpy_array(self.data_dir / self.config.subjects_path)
        self.logger.info(f"subjects.shape: {self.subjects}")
        self.y = np.argmax(self.y_onehot, axis=1)
        self.sample_split_codes = np.full(len(self.y), -1, dtype=np.int8)
        self.split_masks = {"all": np.ones(len(self.y), dtype=bool)}

    @staticmethod
    def _candidate_array_paths(path: Path) -> tuple[Path, ...]:
        if path.suffix in {".npy", ".npz"}:
            alternate_suffix = ".npz" if path.suffix == ".npy" else ".npy"
            return (path, path.with_suffix(alternate_suffix))
        return (path, path.with_suffix(".npz"), path.with_suffix(".npy"))

    @classmethod
    def _resolve_array_path(cls, path: Path) -> Path:
        for candidate in cls._candidate_array_paths(path):
            if candidate.is_file():
                return candidate
        candidates = ", ".join(
            str(candidate) for candidate in cls._candidate_array_paths(path)
        )
        raise FileNotFoundError(f"Could not find numpy array file. Tried: {candidates}")

    @classmethod
    def _load_numpy_array(cls, path: Path) -> np.ndarray:
        """Load an array saved as .npy or compressed .npz.

        For .npz files, the converter in scripts/ stores arrays under the key
        "data". Single-array archives without that key are also accepted.
        """
        resolved_path = cls._resolve_array_path(path)
        loaded = np.load(resolved_path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded:
                if "data" in loaded.files:
                    return loaded["data"]
                if len(loaded.files) == 1:
                    return loaded[loaded.files[0]]
                raise ValueError(
                    f"Expected key 'data' or a single array in {resolved_path}; "
                    f"found keys={loaded.files}"
                )
        return loaded

    def _resolve_biovid_part_dir(self) -> Path:
        """Resolve the BioVid PartA directory from common repository layouts."""
        candidates = (
            self.data_dir / "BioVid" / self.config.biovid_part_dir,
            self.data_dir / self.config.biovid_part_dir,
            self.data_dir,
        )
        for candidate in candidates:
            if (candidate / self.config.biovid_train_split_dir).is_dir() and (
                candidate / self.config.biovid_test_split_dir
            ).is_dir():
                return candidate
        raise FileNotFoundError(
            "Could not resolve BioVid PartA directory with Train/Test subfolders from "
            f"data_dir={self.data_dir!s}"
        )

    @staticmethod
    def _subject_key_from_data_filename(path: Path) -> str:
        """Convert '<subject>_data.npy/.npz' to '<subject>'."""
        if path.suffix not in {".npy", ".npz"}:
            raise ValueError(f"Unexpected data filename extension: {path.name}")
        suffix = "_data"
        if not path.stem.endswith(suffix):
            raise ValueError(f"Unexpected data filename format: {path.name}")
        return path.stem[: -len(suffix)]

    @staticmethod
    def _subject_key_from_label_filename(path: Path) -> str:
        """Convert '<subject>_label.npy/.npz' to '<subject>'."""
        if path.suffix not in {".npy", ".npz"}:
            raise ValueError(f"Unexpected label filename extension: {path.name}")
        suffix = "_label"
        if not path.stem.endswith(suffix):
            raise ValueError(f"Unexpected label filename format: {path.name}")
        return path.stem[: -len(suffix)]

    @staticmethod
    def _prefer_npz(paths: List[Path]) -> Path:
        return sorted(paths, key=lambda path: (path.suffix != ".npz", path.name))[0]

    @classmethod
    def _collect_biovid_files(cls, modality_dir: Path, file_kind: str) -> Dict[str, Path]:
        paths = sorted(
            [
                path
                for path in (
                    list(modality_dir.glob(f"*_{file_kind}.npy"))
                    + list(modality_dir.glob(f"*_{file_kind}.npz"))
                )
                if not path.name.startswith(".")
            ]
        )
        grouped_paths: Dict[str, List[Path]] = {}
        for path in paths:
            if file_kind == "data":
                subject_key = cls._subject_key_from_data_filename(path)
            elif file_kind == "label":
                subject_key = cls._subject_key_from_label_filename(path)
            else:
                raise ValueError(f"Unsupported BioVid file kind: {file_kind}")
            grouped_paths.setdefault(subject_key, []).append(path)

        return {
            subject_key: cls._prefer_npz(subject_paths)
            for subject_key, subject_paths in grouped_paths.items()
        }

    def _load_biovid_part_a_data(self) -> None:
        """Load BioVid Part A pre-segmented train/test files."""
        part_dir = self._resolve_biovid_part_dir()
        self.logger.info(f"Resolved BioVid Part A directory: {part_dir}")
        modalities = tuple(self.config.biovid_modalities)
        split_dir_names = {
            "train": self.config.biovid_train_split_dir,
            "test": self.config.biovid_test_split_dir,
        }

        split_maps: Dict[str, Dict[str, Dict[str, Path]]] = {}
        all_subject_keys = set()

        for split_name, split_dir_name in split_dir_names.items():
            split_dir = part_dir / split_dir_name
            modality_data_maps: Dict[str, Dict[str, Path]] = {}
            modality_label_maps: Dict[str, Dict[str, Path]] = {}

            for modality in modalities:
                modality_dir = split_dir / modality
                if not modality_dir.is_dir():
                    raise FileNotFoundError(
                        f"Expected modality directory missing: {modality_dir}"
                    )

                data_map = self._collect_biovid_files(modality_dir, "data")
                label_map = self._collect_biovid_files(modality_dir, "label")

                missing_label = sorted(set(data_map) - set(label_map))
                missing_data = sorted(set(label_map) - set(data_map))
                if missing_label or missing_data:
                    raise ValueError(
                        f"Mismatched data/label files for split={split_name}, modality={modality}, "
                        f"missing_label={missing_label[:3]}, missing_data={missing_data[:3]}"
                    )

                modality_data_maps[modality] = data_map
                modality_label_maps[modality] = label_map

            common_subjects = sorted(
                set.intersection(
                    *(
                        set(modality_data_maps[modality].keys())
                        for modality in modalities
                    )
                )
            )
            if not common_subjects:
                raise ValueError(f"No subjects found for split={split_name}")

            split_maps[split_name] = {
                "data": {
                    modality: {
                        subject: modality_data_maps[modality][subject]
                        for subject in common_subjects
                    }
                    for modality in modalities
                },
                "labels": {
                    modality: {
                        subject: modality_label_maps[modality][subject]
                        for subject in common_subjects
                    }
                    for modality in modalities
                },
            }
            all_subject_keys.update(common_subjects)

        if not all_subject_keys:
            raise ValueError("BioVid Part A contains no subjects.")

        sorted_subject_keys = sorted(all_subject_keys)
        self.biovid_subject_key_to_int = {
            subject_key: idx for idx, subject_key in enumerate(sorted_subject_keys)
        }
        self.biovid_subject_int_to_key = {
            idx: subject_key
            for subject_key, idx in self.biovid_subject_key_to_int.items()
        }

        X_rows = []
        y_rows = []
        subject_rows = []
        split_rows = []

        for split_name in ("train", "test"):
            split_code = self.split_name_to_code[split_name]
            split_data_maps = split_maps[split_name]["data"]
            split_label_maps = split_maps[split_name]["labels"]
            split_subjects = sorted(split_data_maps[modalities[0]].keys())

            for subject_key in split_subjects:
                modality_arrays = []
                reference_labels = None
                for modality in modalities:
                    data_array = self._load_numpy_array(
                        split_data_maps[modality][subject_key]
                    )
                    if data_array.ndim == 2:
                        data_array = data_array[..., np.newaxis]
                    if data_array.ndim != 3 or data_array.shape[-1] != 1:
                        raise ValueError(
                            f"Expected shape [n_samples, seq_len, 1] for "
                            f"split={split_name}, subject={subject_key}, modality={modality}, "
                            f"got {data_array.shape}"
                        )
                    modality_arrays.append(data_array.astype(np.float32, copy=False))

                    labels = self._load_numpy_array(
                        split_label_maps[modality][subject_key]
                    ).reshape(-1)
                    labels = labels.astype(np.int32, copy=False)
                    if reference_labels is None:
                        reference_labels = labels
                    elif not np.array_equal(reference_labels, labels):
                        raise ValueError(
                            f"Label mismatch across modalities for split={split_name}, "
                            f"subject={subject_key}"
                        )

                subject_X = np.concatenate(modality_arrays, axis=2).astype(
                    np.float32, copy=False
                )
                subject_y = reference_labels
                if subject_X.shape[0] != subject_y.shape[0]:
                    raise ValueError(
                        f"Sample/label count mismatch for split={split_name}, subject={subject_key}: "
                        f"{subject_X.shape[0]} != {subject_y.shape[0]}"
                    )

                subject_id = int(self.biovid_subject_key_to_int[subject_key])
                X_rows.append(subject_X)
                y_rows.append(subject_y)
                subject_rows.append(
                    np.full(subject_X.shape[0], subject_id, dtype=np.int32)
                )
                split_rows.append(
                    np.full(subject_X.shape[0], split_code, dtype=np.int8)
                )

        self.X = np.concatenate(X_rows, axis=0)
        self.y = np.concatenate(y_rows, axis=0)
        self.subjects = np.concatenate(subject_rows, axis=0)
        self.sample_split_codes = np.concatenate(split_rows, axis=0)
        self.y_onehot = None

        self.split_masks = {
            "train": self.sample_split_codes == self.split_name_to_code["train"],
            "test": self.sample_split_codes == self.split_name_to_code["test"],
        }
        self.config.sequence_length = int(self.X.shape[1])
        self.config.num_sensors = int(self.X.shape[2])

    def _build_index(self):
        """Build index mapping (subject, class) -> sample indices."""
        split_names = ["all"]
        if self.has_predefined_split:
            split_names = ["all", "train", "test"]

        self.base_index_by_split = {}
        self.index_by_split = {}
        for split_name in split_names:
            split_mask = self.split_masks.get(split_name)
            if split_mask is None:
                raise ValueError(f"Missing split mask for split='{split_name}'")
            base_index = self._build_base_index_for_mask(split_mask)
            self.base_index_by_split[split_name] = base_index
            if self.window_shift_enabled:
                self.index_by_split[split_name] = self._build_window_shift_index(
                    base_index
                )
            else:
                self.index_by_split[split_name] = base_index

        self.base_index = self.base_index_by_split["all"]
        self.index = self.index_by_split["all"]
        if self.window_shift_enabled:
            self._log_window_shift_summary()

        self._verify_index()

    def _build_base_index_for_mask(
        self, split_mask: np.ndarray
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """Build base index mapping for a specific sample mask."""
        base_index: Dict[int, Dict[int, np.ndarray]] = {}
        for subject in self.unique_subjects:
            base_index[subject] = {}
            subject_mask = (self.subjects == subject) & split_mask
            for episodic_class_id, raw_class_id in enumerate(self.task_class_ids):
                class_mask = self.y == raw_class_id
                combined_mask = subject_mask & class_mask
                indices = np.where(combined_mask)[0]
                base_index[subject][episodic_class_id] = indices
        return base_index

    def _get_split_mask(self, split: str) -> np.ndarray:
        """Return sample mask for split ('all', 'train', 'test')."""
        normalized_split = split.lower()
        if normalized_split not in self.split_masks:
            available = ", ".join(sorted(self.split_masks.keys()))
            raise ValueError(f"Unknown split '{split}'. Available splits: {available}")
        return self.split_masks[normalized_split]

    def _get_index_for_split(self, split: str) -> Dict[int, Dict[int, np.ndarray]]:
        """Return index mapping for the requested split."""
        normalized_split = split.lower()
        if normalized_split not in self.index_by_split:
            available = ", ".join(sorted(self.index_by_split.keys()))
            raise ValueError(
                f"Unknown split '{split}'. Available indexed splits: {available}"
            )
        return self.index_by_split[normalized_split]

    def _get_sampling_index_for_split(
        self, split: str, use_base_index: bool = False
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """Return index for sampling, optionally using non-augmented base entries."""
        if use_base_index:
            return self._get_base_index_for_split(split)
        return self._get_index_for_split(split)

    def _get_base_index_for_split(self, split: str) -> Dict[int, Dict[int, np.ndarray]]:
        """Return non-windowed base index mapping for the requested split."""
        normalized_split = split.lower()
        if normalized_split not in self.base_index_by_split:
            available = ", ".join(sorted(self.base_index_by_split.keys()))
            raise ValueError(
                f"Unknown split '{split}'. Available base-index splits: {available}"
            )
        return self.base_index_by_split[normalized_split]

    def _build_window_shift_index(
        self, base_index: Dict[int, Dict[int, np.ndarray]]
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """Expand index entries into [sample_idx, window_start_idx] references."""
        if self.X.ndim != 3:
            raise ValueError(
                f"Expected rank-3 input [n_samples, seq_len, n_sensors], got {self.X.shape}"
            )

        sampling_rate_hz = int(self.config.sampling_rate_hz)
        raw_sequence_length = int(self.X.shape[1])
        window_length_samples = int(
            round(float(self.config.window_shift_window_seconds) * sampling_rate_hz)
        )
        step_samples = int(
            round(float(self.config.window_shift_step_seconds) * sampling_rate_hz)
        )
        start_min_idx = int(
            round(float(self.config.window_shift_start_min_seconds) * sampling_rate_hz)
        )
        start_max_idx = int(
            round(float(self.config.window_shift_start_max_seconds) * sampling_rate_hz)
        )

        if window_length_samples <= 0:
            raise ValueError("Resolved window length in samples must be > 0.")
        if step_samples <= 0:
            raise ValueError("Resolved window shift step in samples must be > 0.")
        if raw_sequence_length < window_length_samples:
            raise ValueError(
                f"Raw sequence length ({raw_sequence_length}) is shorter than configured "
                f"window length ({window_length_samples})."
            )

        max_valid_start = raw_sequence_length - window_length_samples
        clipped_min_idx = max(0, start_min_idx)
        clipped_max_idx = min(start_max_idx, max_valid_start)
        if clipped_max_idx < clipped_min_idx:
            raise ValueError(
                "No valid window start positions after clipping to signal boundaries. "
                f"Requested start range [{start_min_idx}, {start_max_idx}] with raw length "
                f"{raw_sequence_length} and window length {window_length_samples}."
            )

        window_start_indices = np.arange(
            clipped_min_idx, clipped_max_idx + 1, step_samples, dtype=np.int64
        )
        if window_start_indices.size == 0:
            raise ValueError("Resolved window_start_indices is empty.")

        self.window_start_indices = window_start_indices
        self.window_length_samples = window_length_samples
        self.window_step_samples = step_samples
        self.window_start_min_idx = int(window_start_indices[0])
        self.window_start_max_idx = int(window_start_indices[-1])
        self.config.sequence_length = window_length_samples

        expanded_index: Dict[int, Dict[int, np.ndarray]] = {}
        for subject in self.unique_subjects:
            expanded_index[subject] = {}
            for class_id in range(self.config.n_way):
                sample_indices = base_index[subject][class_id]
                if sample_indices.size == 0:
                    expanded_index[subject][class_id] = np.empty((0, 2), dtype=np.int64)
                    continue
                repeated_sample_indices = np.repeat(
                    sample_indices, window_start_indices.size
                )
                repeated_starts = np.tile(window_start_indices, sample_indices.size)
                refs = np.column_stack(
                    (repeated_sample_indices, repeated_starts)
                ).astype(np.int64, copy=False)
                expanded_index[subject][class_id] = refs

        return expanded_index

    def _extract_windows(self, refs: np.ndarray) -> np.ndarray:
        """Extract fixed windows from [sample_idx, start_idx] references."""
        if refs.size == 0:
            return np.empty(
                (0, self.window_length_samples, self.X.shape[2]), dtype=self.X.dtype
            )

        if refs.ndim != 2 or refs.shape[1] != 2:
            raise ValueError(
                "Windowed references must be shaped [n_refs, 2] as [sample_idx, start_idx]."
            )

        sample_indices = refs[:, 0].astype(np.int64, copy=False)
        start_indices = refs[:, 1].astype(np.int64, copy=False)
        windows = np.empty(
            (len(sample_indices), self.window_length_samples, self.X.shape[2]),
            dtype=self.X.dtype,
        )
        for i, (sample_idx, start_idx) in enumerate(zip(sample_indices, start_indices)):
            windows[i] = self.X[
                sample_idx, start_idx : start_idx + self.window_length_samples, :
            ]
        return windows

    def _gather_samples(self, refs_or_indices: np.ndarray) -> np.ndarray:
        """Gather samples from either base indices [n] or window refs [n, 2]."""
        if refs_or_indices.size == 0:
            if self.window_shift_enabled:
                seq_len = self.window_length_samples
            elif refs_or_indices.ndim == 1:
                seq_len = self.X.shape[1]
            else:
                seq_len = self.window_length_samples
            return np.empty((0, seq_len, self.X.shape[2]), dtype=self.X.dtype)

        if refs_or_indices.ndim == 2:
            return self._extract_windows(refs_or_indices)

        if self.window_shift_enabled:
            start_idx = int(self.window_start_min_idx)
            end_idx = start_idx + int(self.window_length_samples)
            return self.X[refs_or_indices, start_idx:end_idx, :]

        return self.X[refs_or_indices]

    def _log_window_shift_summary(self) -> None:
        """Log global augmentation metadata and class-wise counts."""
        if not self.window_shift_enabled:
            return

        sampling_rate = float(self.config.sampling_rate_hz)
        start_min_sec = self.window_start_min_idx / sampling_rate
        start_max_sec = self.window_start_max_idx / sampling_rate
        step_sec = self.window_step_samples / sampling_rate
        window_sec = self.window_length_samples / sampling_rate
        num_windows = int(self.window_start_indices.size)

        self.logger.info("Window shift augmentation enabled")
        self.logger.info(
            "  Window config: "
            f"window={window_sec:.2f}s ({self.window_length_samples} samples), "
            f"start_range={start_min_sec:.2f}s..{start_max_sec:.2f}s "
            f"([{self.window_start_min_idx}, {self.window_start_max_idx}] samples), "
            f"step={step_sec:.2f}s ({self.window_step_samples} samples), "
            f"windows_per_signal={num_windows}"
        )

        total_original = 0
        total_augmented = 0
        for class_id in range(self.config.n_way):
            original_count = int(
                sum(
                    len(self.base_index[subject][class_id])
                    for subject in self.unique_subjects
                )
            )
            augmented_count = int(
                sum(
                    len(self.index[subject][class_id])
                    for subject in self.unique_subjects
                )
            )
            created_count = augmented_count - original_count
            total_original += original_count
            total_augmented += augmented_count
            self.logger.info(
                f"  Class {class_id}: original={original_count}, "
                f"new={created_count}, total={augmented_count}"
            )

        self.logger.info(
            f"  Overall: original={total_original}, new={total_augmented - total_original}, "
            f"total={total_augmented}"
        )

    def log_window_shift_split_summary(
        self, split_name: str, subjects: List[int], split: str = "all"
    ) -> None:
        """Log augmentation counts for a split's subject set."""
        if not self.window_shift_enabled:
            return

        selected_subjects = [int(subject) for subject in subjects]
        if not selected_subjects:
            self.logger.info(f"Window shift [{split_name}] has no subjects.")
            return

        base_index = self._get_base_index_for_split(split)
        index = self._get_index_for_split(split)
        total_original = 0
        total_augmented = 0
        self.logger.info(
            f"Window shift [{split_name}] split={split} subjects={selected_subjects} "
            f"(n={len(selected_subjects)})"
        )
        for class_id in range(self.config.n_way):
            original_count = int(
                sum(len(base_index[subject][class_id]) for subject in selected_subjects)
            )
            augmented_count = int(
                sum(len(index[subject][class_id]) for subject in selected_subjects)
            )
            created_count = augmented_count - original_count
            total_original += original_count
            total_augmented += augmented_count
            self.logger.info(
                f"  [{split_name}] class {class_id}: "
                f"original={original_count}, new={created_count}, total={augmented_count}"
            )

        self.logger.info(
            f"  [{split_name}] overall: original={total_original}, "
            f"new={total_augmented - total_original}, total={total_augmented}"
        )

    def _verify_index(self):
        """Verify that the index is valid for sampling."""
        for split_name, split_index in self.index_by_split.items():
            min_samples_per_class = float("inf")
            for subject in self.unique_subjects:
                for episodic_class_id, raw_class_id in enumerate(self.task_class_ids):
                    n_samples = len(split_index[subject][episodic_class_id])
                    min_samples_per_class = min(min_samples_per_class, n_samples)
                    if n_samples < self.config.k_shot + self.config.q_query:
                        warnings.warn(
                            f"Split {split_name}, subject {subject}, raw class {raw_class_id} "
                            f"has only {n_samples} samples, but "
                            f"{self.config.k_shot + self.config.q_query} are needed for sampling."
                        )
            self.logger.info(
                f"  Minimum samples per (subject, class) in split={split_name}: "
                f"{min_samples_per_class}"
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

    def compute_split_normalization_stats(
        self, subjects: List[int], split: str = "all"
    ) -> Dict[str, np.ndarray]:
        """Compute modality-wise stats over all samples/windows in a LOSO split."""
        selected_subjects = [int(subject) for subject in subjects]
        if not selected_subjects:
            raise ValueError("subjects must contain at least one subject id")
        split_mask = self._get_split_mask(split)

        sum_values = np.zeros((self.X.shape[2],), dtype=np.float64)
        sum_square_values = np.zeros((self.X.shape[2],), dtype=np.float64)
        count = 0

        for subject in selected_subjects:
            subject_indices = np.where((self.subjects == subject) & split_mask)[0]
            if subject_indices.size == 0:
                continue

            if self.window_shift_enabled:
                for start_idx in self.window_start_indices:
                    windows = self.X[
                        subject_indices,
                        start_idx : start_idx + self.window_length_samples,
                        :,
                    ]
                    sum_values += np.sum(windows, axis=(0, 1))
                    sum_square_values += np.sum(np.square(windows), axis=(0, 1))
                    count += windows.shape[0] * windows.shape[1]
            else:
                subject_data = self.X[subject_indices]
                sum_values += np.sum(subject_data, axis=(0, 1))
                sum_square_values += np.sum(np.square(subject_data), axis=(0, 1))
                count += subject_data.shape[0] * subject_data.shape[1]

        if count == 0:
            raise ValueError(
                "No samples available to compute split normalization stats"
            )

        mean = sum_values / count
        variance = np.maximum((sum_square_values / count) - np.square(mean), 0.0)
        std = np.sqrt(variance) + 1e-8
        return {
            "mean": mean.reshape(1, 1, -1).astype(self.X.dtype, copy=False),
            "std": std.reshape(1, 1, -1).astype(self.X.dtype, copy=False),
        }

    def get_subject_data(
        self, subject: int, normalize: bool = True, split: str = "all"
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
        mask = (self.subjects == subject) & self._get_split_mask(split)
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
        split_normalization_stats: Optional[Dict[str, np.ndarray]] = None,
        split: str = "all",
        use_base_index: bool = False,
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
                - 'split': normalize with precomputed split-level stats
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
        k_shot = k_shot or self.config.k_shot
        q_query = q_query or self.config.q_query
        subject = int(subject)

        if rng is not None:
            local_rng = rng
        elif seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = np.random.default_rng()

        split_index = self._get_sampling_index_for_split(
            split,
            use_base_index=use_base_index,
        )

        support_X, support_y = [], []
        query_X, query_y = [], []
        support_subjects, query_subjects = [], []

        for class_id in range(self.config.n_way):
            indices = split_index[subject][class_id]
            total_to_sample = self._resolve_total_samples_to_draw(
                available_count=len(indices),
                k_shot=k_shot,
                q_query=q_query,
                allow_partial_query=allow_partial_query,
            )
            sampled_positions = local_rng.choice(
                len(indices), size=total_to_sample, replace=False
            )
            sampled_indices = indices[sampled_positions]
            support_idx = sampled_indices[:k_shot]
            query_idx = sampled_indices[k_shot:]

            support_X.append(self._gather_samples(support_idx))
            support_y.append(np.full(k_shot, class_id, dtype=np.int32))
            query_X.append(self._gather_samples(query_idx))
            query_y.append(np.full(len(query_idx), class_id, dtype=np.int32))

            if include_sample_subjects:
                support_subjects.append(np.full(k_shot, subject, dtype=np.int32))
                query_subjects.append(np.full(len(query_idx), subject, dtype=np.int32))

        support_X = np.concatenate(support_X, axis=0)
        support_y = np.concatenate(support_y, axis=0)
        query_X = np.concatenate(query_X, axis=0)
        query_y = np.concatenate(query_y, axis=0)

        if normalize_mode == "subject":
            support_X = self._normalize_data(support_X, subject)
            query_X = self._normalize_data(query_X, subject)
        elif normalize_mode == "split":
            stats = split_normalization_stats
            if stats is None:
                stats = self.compute_split_normalization_stats([subject], split=split)
            support_X = self._apply_stats(support_X, stats)
            query_X = self._apply_stats(query_X, stats)
        elif normalize_mode == "support":
            stats = self._compute_batch_stats(support_X)
            support_X = self._apply_stats(support_X, stats)
            query_X = self._apply_stats(query_X, stats)
        elif normalize_mode == "none":
            pass
        else:
            raise ValueError(
                f"Unknown normalize_mode: {normalize_mode}. Use 'subject', 'split', 'support', or 'none'."
            )

        support_perm = local_rng.permutation(len(support_y))
        query_perm = local_rng.permutation(len(query_y))
        task = {
            "support_X": support_X[support_perm],
            "support_y": support_y[support_perm],
            "query_X": query_X[query_perm],
            "query_y": query_y[query_perm],
            "subject": subject,
        }
        if include_sample_subjects:
            task["support_subjects"] = np.concatenate(support_subjects, axis=0)[
                support_perm
            ]
            task["query_subjects"] = np.concatenate(query_subjects, axis=0)[query_perm]
        return task

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
        split_normalization_stats: Optional[Dict[str, np.ndarray]] = None,
        split: str = "all",
        use_base_index: bool = False,
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

        split_index = self._get_sampling_index_for_split(
            split,
            use_base_index=use_base_index,
        )
        support_X, support_y, support_subjects = [], [], []
        query_X, query_y, query_subjects = [], [], []

        for class_id in range(self.config.n_way):
            pooled_indices = []
            pooled_subject_ids = []
            for subject in selected_subjects:
                indices = split_index[subject][class_id]
                pooled_indices.append(indices)
                pooled_subject_ids.extend([subject] * len(indices))

            pooled_indices = np.concatenate(pooled_indices, axis=0)

            total_to_sample = self._resolve_total_samples_to_draw(
                available_count=len(pooled_indices),
                k_shot=k_shot,
                q_query=q_query,
                allow_partial_query=allow_partial_query,
            )
            sampled_positions = local_rng.choice(
                len(pooled_indices), size=total_to_sample, replace=False
            )
            pooled_subject_ids = np.asarray(pooled_subject_ids, dtype=np.int32)

            sampled_indices = pooled_indices[sampled_positions]
            sampled_subject_ids = pooled_subject_ids[sampled_positions]
            support_idx = sampled_indices[:k_shot]
            query_idx = sampled_indices[k_shot:]
            support_subject_ids = sampled_subject_ids[:k_shot]
            query_subject_ids = sampled_subject_ids[k_shot:]

            support_X.append(self._gather_samples(support_idx))
            support_y.append(np.full(k_shot, class_id, dtype=np.int32))
            support_subjects.append(support_subject_ids)
            query_X.append(self._gather_samples(query_idx))
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
        elif normalize_mode == "split":
            stats = split_normalization_stats
            if stats is None:
                stats = self.compute_split_normalization_stats(
                    selected_subjects,
                    split=split,
                )
            support_X = self._apply_stats(support_X, stats)
            query_X = self._apply_stats(query_X, stats)
        elif normalize_mode == "support":
            stats = self._compute_batch_stats(support_X)
            support_X = self._apply_stats(support_X, stats)
            query_X = self._apply_stats(query_X, stats)
        elif normalize_mode == "none":
            pass
        else:
            raise ValueError(
                f"Unknown normalize_mode: {normalize_mode}. Use 'subject', 'split', 'support', or 'none'."
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
        split_normalization_stats: Optional[Dict[str, np.ndarray]] = None,
        split: str = "all",
        use_base_index: bool = False,
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

        split_index = self._get_sampling_index_for_split(
            split,
            use_base_index=use_base_index,
        )
        support_X, support_y, support_subjects = [], [], []
        query_X, query_y, query_subjects = [], [], []

        for class_id in range(self.config.n_way):
            support_indices = split_index[support_subject][class_id]
            query_indices = split_index[query_subject][class_id]

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

            support_positions = local_rng.choice(
                len(support_indices), size=k_shot, replace=False
            )
            query_positions = local_rng.choice(
                len(query_indices), size=q_query, replace=False
            )
            sampled_support_idx = support_indices[support_positions]
            sampled_query_idx = query_indices[query_positions]

            support_X.append(self._gather_samples(sampled_support_idx))
            support_y.append(np.full(k_shot, class_id, dtype=np.int32))
            support_subjects.append(np.full(k_shot, support_subject, dtype=np.int32))
            query_X.append(self._gather_samples(sampled_query_idx))
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
        elif normalize_mode == "split":
            if split_normalization_stats is None:
                stats_subjects = sorted({support_subject, query_subject})
                split_normalization_stats = self.compute_split_normalization_stats(
                    stats_subjects,
                    split=split,
                )
            support_X = self._apply_stats(support_X, split_normalization_stats)
            query_X = self._apply_stats(query_X, split_normalization_stats)
        elif normalize_mode == "support":
            stats = self._compute_batch_stats(support_X)
            support_X = self._apply_stats(support_X, stats)
            query_X = self._apply_stats(query_X, stats)
        elif normalize_mode == "none":
            pass
        else:
            raise ValueError(
                f"Unknown normalize_mode: {normalize_mode}. Use 'subject', 'split', 'support', or 'none'."
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

    def sample_task_mixed_subject_pools(
        self,
        support_subjects: List[int],
        query_subjects: List[int],
        k_shot: Optional[int] = None,
        q_query: Optional[int] = None,
        seed: Optional[int] = None,
        normalize_mode: str = "subject",
        rng: Optional[np.random.Generator] = None,
        include_sample_subjects: bool = False,
        split_normalization_stats: Optional[Dict[str, np.ndarray]] = None,
        split: str = "all",
        use_base_index: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Sample one task where support/query are drawn from disjoint subject pools.

        Args:
            support_subjects: Subject IDs eligible for support sampling
            query_subjects: Subject IDs eligible for query sampling (must be disjoint)
            k_shot: Number of support samples per class
            q_query: Number of query samples per class
            seed: Random seed
            normalize_mode: 'subject', 'split', 'support', or 'none'
            rng: Optional numpy Generator
            include_sample_subjects: Whether to include source subject ids in output
            split_normalization_stats: Optional precomputed stats for normalize_mode='split'
            split: Data split selector ('all', 'train', 'test')
            use_base_index: If True, sample from non-augmented base references
        """
        support_subjects = [int(subject) for subject in support_subjects]
        query_subjects = [int(subject) for subject in query_subjects]
        if not support_subjects:
            raise ValueError("support_subjects must contain at least one subject id")
        if not query_subjects:
            raise ValueError("query_subjects must contain at least one subject id")
        overlap = sorted(set(support_subjects).intersection(query_subjects))
        if overlap:
            raise ValueError(
                f"support_subjects and query_subjects must be disjoint, overlap={overlap}"
            )

        k_shot = k_shot or self.config.k_shot
        q_query = q_query or self.config.q_query
        if rng is not None:
            local_rng = rng
        elif seed is not None:
            local_rng = np.random.default_rng(seed)
        else:
            local_rng = np.random.default_rng()

        split_index = self._get_sampling_index_for_split(
            split,
            use_base_index=use_base_index,
        )
        support_X, support_y, support_subject_ids_out = [], [], []
        query_X, query_y, query_subject_ids_out = [], [], []

        for class_id in range(self.config.n_way):
            support_pool = []
            support_pool_subjects = []
            for subject in support_subjects:
                refs = split_index[subject][class_id]
                support_pool.append(refs)
                support_pool_subjects.extend([subject] * len(refs))
            support_pool = np.concatenate(support_pool, axis=0)
            if len(support_pool) < k_shot:
                raise ValueError(
                    f"Support pool has only {len(support_pool)} samples for class {class_id}, "
                    f"but k_shot={k_shot}."
                )
            support_positions = local_rng.choice(
                len(support_pool), size=k_shot, replace=False
            )
            support_pool_subjects = np.asarray(support_pool_subjects, dtype=np.int32)
            sampled_support_refs = support_pool[support_positions]
            sampled_support_subjects = support_pool_subjects[support_positions]

            query_pool = []
            query_pool_subjects = []
            for subject in query_subjects:
                refs = split_index[subject][class_id]
                query_pool.append(refs)
                query_pool_subjects.extend([subject] * len(refs))
            query_pool = np.concatenate(query_pool, axis=0)
            if len(query_pool) < q_query:
                raise ValueError(
                    f"Query pool has only {len(query_pool)} samples for class {class_id}, "
                    f"but q_query={q_query}."
                )
            query_positions = local_rng.choice(
                len(query_pool), size=q_query, replace=False
            )
            query_pool_subjects = np.asarray(query_pool_subjects, dtype=np.int32)
            sampled_query_refs = query_pool[query_positions]
            sampled_query_subjects = query_pool_subjects[query_positions]

            support_X.append(self._gather_samples(sampled_support_refs))
            support_y.append(np.full(k_shot, class_id, dtype=np.int32))
            support_subject_ids_out.append(sampled_support_subjects)
            query_X.append(self._gather_samples(sampled_query_refs))
            query_y.append(np.full(q_query, class_id, dtype=np.int32))
            query_subject_ids_out.append(sampled_query_subjects)

        support_X = np.concatenate(support_X, axis=0)
        support_y = np.concatenate(support_y, axis=0)
        support_subject_ids_out = np.concatenate(support_subject_ids_out, axis=0)
        query_X = np.concatenate(query_X, axis=0)
        query_y = np.concatenate(query_y, axis=0)
        query_subject_ids_out = np.concatenate(query_subject_ids_out, axis=0)

        if normalize_mode == "subject":
            support_X = self._normalize_data_by_subjects(
                support_X, support_subject_ids_out
            )
            query_X = self._normalize_data_by_subjects(query_X, query_subject_ids_out)
        elif normalize_mode == "split":
            if split_normalization_stats is None:
                stats_subjects = sorted(set(support_subjects).union(query_subjects))
                split_normalization_stats = self.compute_split_normalization_stats(
                    stats_subjects,
                    split=split,
                )
            support_X = self._apply_stats(support_X, split_normalization_stats)
            query_X = self._apply_stats(query_X, split_normalization_stats)
        elif normalize_mode == "support":
            stats = self._compute_batch_stats(support_X)
            support_X = self._apply_stats(support_X, stats)
            query_X = self._apply_stats(query_X, stats)
        elif normalize_mode == "none":
            pass
        else:
            raise ValueError(
                f"Unknown normalize_mode: {normalize_mode}. Use 'subject', 'split', 'support', or 'none'."
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
            task["support_subjects"] = support_subject_ids_out[support_perm]
            task["query_subjects"] = query_subject_ids_out[query_perm]
        return task

    def sample_meta_task_batch(
        self,
        subjects: List[int],
        batch_size: int,
        k_shot: Optional[int] = None,
        q_query: Optional[int] = None,
        split: str = "all",
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
        return [
            self.sample_task(s, k_shot, q_query, split=split) for s in sampled_subjects
        ]

    def get_split_subjects(self, split: str = "all") -> List[int]:
        """Return sorted subject IDs that have samples in a split."""
        normalized_split = split.lower()
        if normalized_split not in self.split_subjects:
            available = ", ".join(sorted(self.split_subjects.keys()))
            raise ValueError(
                f"Unknown split '{split}'. Available split subjects: {available}"
            )
        return list(self.split_subjects[normalized_split])

    def leave_one_subject_out_split(self, test_subject: int) -> Tuple[List[int], int]:
        """
        Create leave-one-subject-out split.

        Args:
            test_subject: Subject ID to hold out for testing

        Returns:
            train_subjects: List of training subject IDs
            test_subject: Held-out subject ID
        """
        if self.has_predefined_split:
            train_subjects = self.get_split_subjects("train")
            return train_subjects, int(test_subject)

        train_subjects = [s for s in self.unique_subjects if s != test_subject]
        return train_subjects, test_subject

    def get_few_shot_split(
        self,
        subject: int,
        k_shot: int,
        seed: Optional[int] = None,
        split: str = "all",
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
        split_index = self._get_index_for_split(split)

        support_X, support_y = [], []
        eval_X, eval_y = [], []

        for class_id in range(self.config.n_way):
            indices = split_index[subject][class_id]
            shuffled_indices = local_rng.permutation(indices)

            support_idx = shuffled_indices[:k_shot]
            eval_idx = shuffled_indices[k_shot:]

            support_X.append(self._gather_samples(support_idx))
            support_y.append(np.full(len(support_idx), class_id, dtype=np.int32))
            eval_X.append(self._gather_samples(eval_idx))
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
