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
    num_tcn_blocks: int = 3  # Number of Temporal Conv Blocks in the Architecture
    filters_list: Optional[List[int]] = (
        8, 16, 32, # Number of Filters in the Convolutional Layers
    )
    strides: int = 2 # Stride in Convolutional Layer
    pooling_size: int = 2 # Pooling in Convolutional Blocks
    embedding_dim: int = 128  # Encoder embedding dimension
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
    supcon_loss_weight: float = 0.5  # Weight for supervised contrastive embedding loss
    supcon_temperature: float = 0.05  # Temperature for supervised contrastive loss
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
        if self.supcon_loss_weight < 0:
            raise ValueError("supcon_loss_weight must be non-negative")
        if self.supcon_temperature <= 0:
            raise ValueError("supcon_temperature must be > 0")
