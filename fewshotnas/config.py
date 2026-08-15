from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class FewShotNASConfig:
    seed: int = 42
    n_trials: int = 100
    max_epochs: int = 5
    tasks_per_epoch: int = 10_000
    task_batch_size: int = 16
    support_repeats: int = 100
    k_shot: int = 10
    q_query: int = 10
    train_subject_count: int = 70
    validation_subject_count: int = 17
    search_patience: int = 2
    max_parameters: int = 8_000_000
    modalities: tuple[str, str] = ("GSR", "ECG")
    raw_class_ids: tuple[int, int] = (0, 4)
    expected_subjects: int = 87

    def __post_init__(self) -> None:
        for name in (
            "n_trials", "max_epochs", "tasks_per_epoch", "task_batch_size",
            "support_repeats", "k_shot", "q_query", "train_subject_count",
            "validation_subject_count", "max_parameters", "expected_subjects",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.train_subject_count + self.validation_subject_count != self.expected_subjects:
            raise ValueError("train and validation subject counts must cover expected_subjects")
        if self.search_patience < 0:
            raise ValueError("search_patience must be >= 0")
        if len(self.modalities) != 2 or len(set(self.modalities)) != 2:
            raise ValueError("CrossMod NAS requires two distinct modalities")
        if self.raw_class_ids != (0, 4):
            raise ValueError("FewShotNAS currently implements BioVid T0-vs-T4")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class FewShotArchitectureSpec:
    crossmod_num_heads: int = 8
    crossmod_hidden_dim: int = 128
    crossmod_num_layers: int = 2
    crossmod_attention_dropout_rate: float = 0.25
    crossmod_ff_activation: str = "relu"
    crossmod_fusion_mode: str = "cross_attention_concat"
    can_meta_depth: int = 1
    can_meta_hidden_dim: int = 32
    can_meta_activation: str = "gelu"
    can_temporal_pooling: str = "gated"
    can_attention_temperature: float = 0.025
    can_local_pool_temperature: float = 0.1
    prototype_feature_normalization: str = "none"
    prototype_aggregation: str = "mean"
    prototype_attention_temperature: float = 0.2
    learned_prototype_slots_per_class: int = 2
    prototype_bank_init_samples_per_class: int = 128
    can_logit_scale_initial: float = 10.0
    learning_rate: float = 6e-4
    lr_decay_alpha: float = 0.1
    can_local_loss_weight: float = 0.25
    can_margin_loss_weight: float = 0.5
    can_margin_target: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FewShotArchitectureSpec":
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__})
