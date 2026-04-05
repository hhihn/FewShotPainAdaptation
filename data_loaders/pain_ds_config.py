from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PainDatasetConfig:
    """Configuration for the pain dataset."""

    # Data dimensions
    num_subjects: int = 52
    num_stimuli_levels: int = 6  # 6 temperature/pain levels
    num_repetitions: int = 8  # 8 repetitions per stimulus level
    sequence_length: int = 2500  # 10 seconds × 250 Hz
    num_sensors: int = 3  # Number of modalities
    num_tcn_blocks: int = 1  # Number of Temporal Conv Blocks in the Architecture
    filters_list: Optional[List[int]] = (
        2, # Number of Filters in the Convolutional Layers
    )
    tcn_dilation_rates: Optional[List[int]] = None  # Dilation rate per TCN block
    tcn_kernel_size: int = 3  # Kernel size used by Conv1D layers in each TCN block
    strides: int = 2 # Stride used by temporal pooling between TCN blocks
    pooling_size: int = 2 # Pool size used between TCN blocks
    tcn_dropout_rate: float = 0.3  # Dropout rate inside the TCN encoder
    embedding_dim: int = 8  # Encoder embedding dimension
    tcn_attention_heads: int = 4  # Number of self-attention heads inside the TCN
    tcn_attention_key_dim: int = 32  # Key dimension per TCN attention head
    tcn_attention_dropout: float = 0.2  # Dropout inside the TCN attention layer
    tcn_attention_pool_size: int = 0  # Downsample factor before self-attention
    fusion_transformer_heads: int = 4  # Heads for transformer-based fusion
    fusion_transformer_layers: int = 2  # Number of transformer fusion layers
    fusion_transformer_ffn_dim: int = 128  # FFN hidden dimension in fusion transformer
    fusion_ib_beta: float = 1e-3  # Information bottleneck KL weight
    clear_session_per_fold: bool = True  # Free TF graph memory between LOSO folds
    single_loso_fold: bool = True  # If True, run only one LOSO fold (testing mode)
    single_loso_test_subject: Optional[int] = None  # Optional explicit held-out subject
    # Sensors used
    painmonit_sensors: Tuple[str] = ("Bvp", "Eda_E4", "Resp", "Eda_RB", "Ecg", "Emg")
    # Modality information
    modality_names: Tuple[str, ...] = (
        "EDA",  # idx 1
        "ECG",  # idx 4
        "EMG",  # idx 5
    )
    # Sensor to index mapping
    sensor_idx = [1, 4, 5]

    # Meta-learning settings
    task_class_ids: Tuple[int, ...] = (
        0,
        5,
    )  # Raw dataset labels included in each task
    n_way: int = len(
        task_class_ids
    )  # Number of classes per task; derived from task_class_ids
    k_shot: int = 3  # Support samples per class
    q_query: int = 3  # Query samples per class
    task_normalize_mode: str = "subject"  # Episodic normalization: subject, support, or none
    classifier_mode: str = "soft_knn"  # Episodic classifier: prototype or soft_knn
    supcon_loss_weight: float = 0.0  # Weight for supervised contrastive embedding loss
    supcon_temperature: float = 0.05  # Temperature for supervised contrastive loss
    triplet_loss_weight: float = 1.0  # Weight for BatchAllTriplet embedding loss
    triplet_margin: float = 0.2  # Margin used by BatchAllTriplet loss
    train_batch_size: int = 20  # Number of tasks per optimizer update
    num_epochs: int = 10  # Number of epochs per fold
    tasks_per_epoch: int = 100  # Number of train tasks sampled per epoch
    val_tasks: int = 20  # Number of validation tasks per validation run
    subject_eval_tasks: int = 20  # Number of held-out subject eval tasks
    k_shot_adaptation_steps: int = 10  # Inner-loop adaptation steps on held-out subject
    train_log_every: int = 10  # Log train metrics every N sampled train tasks
    eval_log_every: int = 5  # Log validation metrics every N sampled train tasks
    val_batch_size: int = 32  # Validation task batch size
    val_every_n_train_steps: int = 20  # Run validation every N processed train batches
    seed: int = 42  # Global seed for reproducible runs
    deterministic_ops: bool = True  # TensorFlow deterministic op mode

    # Data paths
    data_path: str = "X_pre.npy"
    labels_path: str = "y_heater.npy"
    subjects_path: str = "subjects.npy"

    def __post_init__(self) -> None:
        self.task_class_ids = tuple(int(class_id) for class_id in self.task_class_ids)
        if not self.task_class_ids:
            raise ValueError("task_class_ids must contain at least one class id")
        if len(set(self.task_class_ids)) != len(self.task_class_ids):
            raise ValueError("task_class_ids must be unique")
        self.n_way = len(self.task_class_ids)
        if self.task_normalize_mode not in {"subject", "support", "none"}:
            raise ValueError(
                "task_normalize_mode must be one of: 'subject', 'support', 'none'"
            )
        if self.classifier_mode not in {"prototype", "soft_knn"}:
            raise ValueError(
                "classifier_mode must be one of: 'prototype', 'soft_knn'"
            )
        if self.supcon_loss_weight < 0:
            raise ValueError("supcon_loss_weight must be non-negative")
        if self.supcon_temperature <= 0:
            raise ValueError("supcon_temperature must be > 0")
        if self.triplet_loss_weight < 0:
            raise ValueError("triplet_loss_weight must be non-negative")
        if self.triplet_margin < 0:
            raise ValueError("triplet_margin must be non-negative")
        if self.tcn_kernel_size <= 0:
            raise ValueError("tcn_kernel_size must be > 0")
        if self.strides <= 0:
            raise ValueError("strides must be > 0")
        if self.pooling_size <= 0:
            raise ValueError("pooling_size must be > 0")
        if self.tcn_dropout_rate < 0 or self.tcn_dropout_rate >= 1:
            raise ValueError("tcn_dropout_rate must be in [0, 1)")
        if self.tcn_attention_heads <= 0:
            raise ValueError("tcn_attention_heads must be > 0")
        if self.tcn_attention_key_dim <= 0:
            raise ValueError("tcn_attention_key_dim must be > 0")
        if self.tcn_attention_dropout < 0 or self.tcn_attention_dropout >= 1:
            raise ValueError("tcn_attention_dropout must be in [0, 1)")
        if self.tcn_attention_pool_size < 0:
            raise ValueError("tcn_attention_pool_size must be >= 0")
        if self.tcn_dilation_rates is not None:
            self.tcn_dilation_rates = [
                int(dilation_rate) for dilation_rate in self.tcn_dilation_rates
            ]
            if len(self.tcn_dilation_rates) != self.num_tcn_blocks:
                raise ValueError(
                    "tcn_dilation_rates length must match num_tcn_blocks"
                )
            if any(dilation_rate <= 0 for dilation_rate in self.tcn_dilation_rates):
                raise ValueError("tcn_dilation_rates values must be > 0")
