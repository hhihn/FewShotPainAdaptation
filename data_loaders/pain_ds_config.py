from typing import Tuple, Optional
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
    embedding_dim: int = 128  # Encoder embedding dimension
    tcn_attention_pool_size: int = 8  # Downsample factor before self-attention
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
    n_way: int = 6  # Number of classes per task (all 6 pain levels)
    k_shot: int = 3  # Support samples per class
    q_query: int = 3  # Query samples per class
    train_batch_size: int = 2  # Number of tasks per optimizer update
    num_epochs: int = 10  # Number of epochs per fold
    tasks_per_epoch: int = 100  # Number of train tasks sampled per epoch
    val_tasks: int = 20  # Number of validation tasks per validation run
    subject_eval_tasks: int = 20  # Number of held-out subject eval tasks
    k_shot_adaptation_steps: int = 10  # Inner-loop adaptation steps on held-out subject
    train_log_every: int = 10  # Log train metrics every N sampled train tasks
    eval_log_every: int = 5  # Log validation metrics every N sampled train tasks
    val_batch_size: int = 32  # Validation task batch size
    val_every_n_train_steps: int = 10  # Run validation every N sampled train tasks
    seed: int = 42  # Global seed for reproducible runs
    deterministic_ops: bool = True  # TensorFlow deterministic op mode

    # Data paths
    data_path: str = "X_pre.npy"
    labels_path: str = "y_heater.npy"
    subjects_path: str = "subjects.npy"
