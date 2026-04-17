#!/usr/bin/env python3
"""Create a mock pain dataset with the same file structure as the real data.

The generated arrays mimic the repository's expected layout:
- X_pre_mock.npy: [n_samples, 2500, 6, 1]
- y_heater_mock.npy: one-hot class labels with 6 classes
- subjects_mock.npy: subject ids per sample

The mock signals are intentionally easy to separate by class:
- within each class, samples are generated from a shared class template plus small noise
- across classes, templates differ in frequency, phase, amplitude, and offsets

This is useful for quick end-to-end sanity checks without using the real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _build_class_template(
    class_id: int,
    sequence_length: int,
    num_modalities: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a class-specific base waveform for all modalities."""
    t = np.linspace(0.0, 1.0, sequence_length, dtype=np.float32)
    template = np.zeros((sequence_length, num_modalities), dtype=np.float32)

    base_freq = 1.0 + 0.35 * class_id
    base_phase = 0.4 * class_id
    base_amp = 0.9 + 0.15 * class_id

    modality_offsets = np.array([0.10, -0.05, 0.25, -0.12, 0.18, -0.08], dtype=np.float32)
    modality_scales = np.array([1.0, 0.7, 1.2, 0.85, 1.1, 0.95], dtype=np.float32)

    for modality_idx in range(num_modalities):
        freq = base_freq + 0.05 * modality_idx
        phase = base_phase + 0.3 * modality_idx
        scale = base_amp * modality_scales[modality_idx]
        carrier = np.sin(2.0 * np.pi * freq * t + phase)
        harmonic = 0.35 * np.cos(2.0 * np.pi * (freq * 2.0) * t + phase / 2.0)
        envelope = 1.0 + 0.1 * np.sin(2.0 * np.pi * (0.5 + 0.1 * class_id) * t)
        template[:, modality_idx] = (
            scale * envelope * (carrier + harmonic) + modality_offsets[modality_idx]
        )

    # Add a tiny class-level pattern so classes are clearly distinct.
    template += rng.normal(loc=0.0, scale=0.02, size=template.shape).astype(np.float32)
    return template


def generate_mock_dataset(
    output_dir: Path = DATA_DIR,
    n_samples: int = 2495,
    sequence_length: int = 2500,
    num_modalities: int = 6,
    num_classes: int = 6,
    num_subjects: int = 52,
    seed: int = 42,
) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_templates = [
        _build_class_template(class_id, sequence_length, num_modalities, rng)
        for class_id in range(num_classes)
    ]

    class_counts = np.full(num_classes, n_samples // num_classes, dtype=int)
    class_counts[: n_samples % num_classes] += 1

    subject_ids = np.arange(num_subjects, dtype=np.int32)
    subject_counts = np.full(num_subjects, n_samples // num_subjects, dtype=int)
    subject_counts[: n_samples % num_subjects] += 1

    X = np.empty((n_samples, sequence_length, num_modalities, 1), dtype=np.float32)
    y = np.zeros((n_samples, num_classes), dtype=np.int32)
    subjects = np.empty((n_samples,), dtype=np.int32)

    sample_idx = 0
    for class_id, class_count in enumerate(class_counts):
        for _ in range(class_count):
            subject_id = int(subject_ids[sample_idx % num_subjects])
            subject_scale = 1.0 + 0.05 * np.sin(0.15 * subject_id)
            subject_shift = 0.02 * subject_id

            sample = class_templates[class_id].copy()
            sample *= subject_scale
            sample += subject_shift

            # Small within-class noise keeps samples similar while preserving class identity.
            sample += rng.normal(loc=0.0, scale=0.04, size=sample.shape).astype(np.float32)
            sample += 0.03 * np.sin(
                np.linspace(0.0, 6.0 * np.pi, sequence_length, dtype=np.float32)
            )[:, None]

            X[sample_idx, :, :, 0] = sample
            y[sample_idx, class_id] = 1
            subjects[sample_idx] = subject_id
            sample_idx += 1

    # Shuffle samples together so subjects and classes are mixed.
    perm = rng.permutation(n_samples)
    X = X[perm]
    y = y[perm]
    subjects = subjects[perm]

    x_path = output_dir / "X_pre_mock.npy"
    y_path = output_dir / "y_heater_mock.npy"
    subjects_path = output_dir / "subjects_mock.npy"

    np.save(x_path, X)
    np.save(y_path, y)
    np.save(subjects_path, subjects)

    return {
        "X_pre": x_path,
        "y_heater": y_path,
        "subjects": subjects_path,
    }


def main() -> None:
    paths = generate_mock_dataset()
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
