"""Configuration for the supervised PainNAS experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class PainNASConfig:
    """Scientific and runtime settings shared by NAS and LOSO."""

    seed: int = 42
    batch_size: int = 40
    n_trials: int = 50
    search_max_epochs: int = 50
    loso_max_epochs: int = 100
    search_patience: int = 8
    loso_patience: int = 15
    search_validation_subjects: int = 17
    max_parameters: int = 32_000_000
    bootstrap_samples: int = 10_000
    dropout_rate: float = 0.25
    num_classes: int = 2
    raw_class_ids: tuple[int, int] = (0, 4)
    modalities: tuple[str, str, str] = ("GSR", "ECG", "EMG")
    expected_sequence_length: int = 1152
    expected_subjects: int = 87

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "batch_size",
            "n_trials",
            "search_max_epochs",
            "loso_max_epochs",
            "search_validation_subjects",
            "max_parameters",
            "bootstrap_samples",
            "num_classes",
            "expected_sequence_length",
            "expected_subjects",
        )
        for field_name in positive_integer_fields:
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be > 0")
        if self.search_patience < 0 or self.loso_patience < 0:
            raise ValueError("early-stopping patience must be >= 0")
        if self.num_classes != 2 or self.raw_class_ids != (0, 4):
            raise ValueError("PainNAS currently implements binary BioVid T0-vs-T4 only")
        if len(self.modalities) != 3:
            raise ValueError("PainNAS requires exactly three physiological modalities")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Return a stable hash used to guard resumable artifacts."""

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


PROTOCOL_WARNING = (
    "Architecture search is performed once using all BioVid subject identities. "
    "The subsequent 87-fold result is exploratory and is not an unbiased nested-LOSO "
    "estimate."
)
