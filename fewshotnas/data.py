from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SubjectSplit:
    train_subjects: tuple[int, ...]
    validation_subjects: tuple[int, ...]
    seed: int

    def validate(self, available: Iterable[int]) -> None:
        known = {int(v) for v in available}
        train, validation = set(self.train_subjects), set(self.validation_subjects)
        if train & validation:
            raise RuntimeError("Training and validation subjects overlap")
        if train | validation != known:
            raise RuntimeError("Subject split does not exactly cover the dataset")


def deterministic_subject_split(
    subjects: Iterable[int], *, train_count: int, validation_count: int, seed: int
) -> SubjectSplit:
    ordered = np.asarray(sorted(int(v) for v in subjects), dtype=np.int32)
    if train_count + validation_count != len(ordered):
        raise ValueError("Requested subject counts do not cover available subjects")
    shuffled = np.random.default_rng(seed).permutation(ordered)
    split = SubjectSplit(
        train_subjects=tuple(sorted(int(v) for v in shuffled[:train_count])),
        validation_subjects=tuple(sorted(int(v) for v in shuffled[train_count:])),
        seed=int(seed),
    )
    split.validate(ordered)
    return split
