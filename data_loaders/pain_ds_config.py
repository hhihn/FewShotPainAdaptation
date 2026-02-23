from typing import Tuple
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
    n_way: int = 6  # Number of classes per episode (all 6 pain levels)
    k_shot: int = 3  # Support samples per class
    q_query: int = 3  # Query samples per class
    seed: int = 42  # Global seed for reproducible runs
    deterministic_ops: bool = True  # TensorFlow deterministic op mode

    # Data paths
    data_path: str = "X_pre.npy"
    labels_path: str = "y_heater.npy"
    subjects_path: str = "subjects.npy"
