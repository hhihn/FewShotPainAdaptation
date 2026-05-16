from typing import Tuple, Optional
from dataclasses import dataclass


SUPPORTED_VALIDATION_CHECKPOINT_METRICS = (
    "loss",
    "task_loss",
    "contrastive_loss",
    "triplet_loss",
    "can_local_loss",
    "can_global_loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "intra_class_similarity",
    "inter_class_similarity",
    "similarity_margin",
)

VALIDATION_CHECKPOINT_MODES = ("auto", "min", "max")


@dataclass
class PainDatasetConfig:
    """Configuration for the pain dataset."""

    # Data dimensions
    num_subjects: int = 52
    num_stimuli_levels: int = 6  # 6 temperature/pain levels
    num_repetitions: int = 8  # 8 repetitions per stimulus level
    sequence_length: int = 2500  # 10 seconds × 250 Hz
    num_sensors: int = 3  # Number of modalities
    eegnet_temporal_filters: int = 8
    eegnet_depth_multiplier: int = 2
    eegnet_separable_filters: int = 16
    eegnet_temporal_kernel_size: int = 64
    eegnet_separable_kernel_size: int = 16
    eegnet_pool_size_1: int = 4
    eegnet_pool_size_2: int = 8
    eegnet_dropout_rate: float = 0.25
    eegnet_l2_weight: float = 1e-4
    embedding_dim: int = 64  # Joint EEGNet encoder embedding dimension
    clear_session_per_fold: bool = True  # Legacy flag; LOSO folds now reuse one graph
    single_loso_fold: bool = True  # If True, run only one LOSO fold (testing mode)
    single_loso_test_subject: Optional[int] = None  # Optional explicit held-out subject
    loso_start_index: Optional[int] = None  # 1-based inclusive LOSO fold start
    loso_stop_index: Optional[int] = None  # 1-based inclusive LOSO fold stop
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
    task_normalize_mode: str = (
        "support"  # Episodic normalization: subject, split, support, or none
    )
    task_construction_mode: str = (
        "single_subject"  # single_subject, cross_subject, or mixed
    )
    classifier_mode: str = "prototype"  # Episodic classifier: prototype or soft_knn
    attention_mode: str = "none"  # none or can
    can_attention_temperature: float = 1.0
    can_meta_hidden_dim: int = 32
    can_local_loss_weight: float = 1.0
    can_global_loss_weight: float = 0.1
    can_transductive_iterations: int = 3
    can_transductive_top_k_per_class: int = 1
    can_transductive_min_confidence: float = 0.0
    triplet_loss_weight: float = 1.0  # Weight for triplet embedding loss
    triplet_margin: float = 0.1  # Margin used by triplet loss
    triplet_mining_strategy: str = (
        "batch_hard"  # batch_hard, batch_all, or triplet_center
    )
    triplet_center_gradient_clip_norm: float = 0.01
    train_batch_size: int = 256  # Number of tasks per optimizer update
    embedding_batch_size: int = (
        1  # Number of tasks encoded together; 1 preserves legacy per-task embedding
    )
    num_epochs: int = 10  # Number of epochs per fold
    tasks_per_epoch: int = 100  # Number of train tasks sampled per epoch
    val_tasks: int = 20  # Number of validation tasks per validation run
    heldout_eval_tasks: int = 20  # Number of held-out evaluation tasks per fold
    subject_eval_tasks: Optional[int] = None  # Deprecated alias for heldout_eval_tasks
    validation_checkpoint_metric: str = (
        "accuracy"  # Validation metric used to select the fold model for held-out eval
    )
    validation_checkpoint_mode: str = (
        "auto"  # auto, min, or max direction for validation checkpoint selection
    )
    lr_schedule: str = "constant"  # Learning-rate schedule: constant or cosine
    lr_decay_alpha: float = 0.1  # Final LR fraction for cosine decay
    k_shot_adaptation_steps: int = 10  # Inner-loop adaptation steps on held-out subject
    train_log_every: int = 10  # Log train metrics every N sampled train tasks
    eval_log_every: int = 5  # Log validation metrics every N sampled train tasks
    val_batch_size: int = 32  # Validation task batch size
    val_every_n_train_steps: int = 20  # Run validation every N processed train batches
    logging_verbosity: int = 1  # 0=minimal, 1=standard, 2=detailed training logs
    train_prefetch_batches: int = 256  # Number of asynchronously prepared train batches
    gradient_clip_norm: Optional[float] = (
        1.0  # Per-gradient norm clip for optimizer updates
    )
    enable_numerics_check: bool = True  # Check train losses/gradients for NaN/Inf
    train_progress_write_every_n_batches: int = (
        10  # Persist train_update CSV rows every N train batches
    )
    csv_flush_every_events: int = 100  # Flush CSV file handle every N written events
    seed: int = 42  # Global seed for reproducible runs
    deterministic_ops: bool = True  # TensorFlow deterministic op mode
    enable_window_shift_augmentation: bool = (
        True  # If True, sample fixed-length shifted windows instead of full signals
    )
    gaussian_noise_std: float = (
        0.01  # Additive Gaussian noise stddev used only during training updates
    )
    window_shift_window_seconds: float = (
        4.0  # Output window duration in seconds (e.g., 4s => 1000 samples @ 250 Hz)
    )
    window_shift_start_min_seconds: float = (
        1.0  # Earliest window start (seconds after signal start)
    )
    window_shift_start_max_seconds: float = (
        6.0  # Latest window start (seconds after signal start)
    )
    window_shift_step_seconds: float = (
        0.2  # Sliding step in seconds (e.g., 0.2s => 50 samples @ 250 Hz)
    )
    sampling_rate_hz: int = 250  # Signal sampling rate used for time->index conversion

    # Data paths
    dataset_source: str = "biovid_part_a"  # painmonit or biovid_part_a
    split_strategy: str = "loso"  # loso or predefined
    data_variant: str = "real"  # real or mock
    data_path: str = "X_pre.npy"
    labels_path: str = "y_heater.npy"
    subjects_path: str = "subjects.npy"
    biovid_part_dir: str = "PartA"
    biovid_train_split_dir: str = "Train"
    biovid_test_split_dir: str = "Test"
    biovid_modalities: Tuple[str, ...] = ("GSR", "ECG", "EMG")

    def __post_init__(self) -> None:
        if self.subject_eval_tasks is not None:
            self.heldout_eval_tasks = int(self.subject_eval_tasks)
        if self.dataset_source not in {"painmonit", "biovid_part_a"}:
            raise ValueError(
                "dataset_source must be one of: 'painmonit', 'biovid_part_a'"
            )
        if self.split_strategy not in {"loso", "predefined"}:
            raise ValueError("split_strategy must be one of: 'loso', 'predefined'")
        if self.dataset_source == "biovid_part_a":
            # BioVid Part A ships with an explicit train/test split.
            self.split_strategy = "predefined"
            self.num_stimuli_levels = 5
            self.sampling_rate_hz = 256
            # Avoid applying an additional sliding-window augmentation on top of
            # BioVid Part A pre-segmented windows.
            self.enable_window_shift_augmentation = False
            if self.task_class_ids == (0, 5):
                self.task_class_ids = (0, 4)
            if self.data_variant == "mock":
                raise ValueError(
                    "data_variant='mock' is only supported for dataset_source='painmonit'"
                )

        self.task_class_ids = tuple(int(class_id) for class_id in self.task_class_ids)
        if not self.task_class_ids:
            raise ValueError("task_class_ids must contain at least one class id")
        if len(set(self.task_class_ids)) != len(self.task_class_ids):
            raise ValueError("task_class_ids must be unique")
        self.n_way = len(self.task_class_ids)
        if self.task_normalize_mode not in {"subject", "split", "support", "none"}:
            raise ValueError(
                "task_normalize_mode must be one of: 'subject', 'split', 'support', 'none'"
            )
        if self.task_construction_mode not in {
            "single_subject",
            "cross_subject",
            "mixed",
        }:
            raise ValueError(
                "task_construction_mode must be one of: 'single_subject', 'cross_subject', 'mixed'"
            )
        if self.classifier_mode not in {"prototype", "soft_knn"}:
            raise ValueError("classifier_mode must be one of: 'prototype', 'soft_knn'")
        self.attention_mode = str(self.attention_mode).strip().lower()
        if self.attention_mode not in {"none", "can"}:
            raise ValueError("attention_mode must be one of: 'none', 'can'")
        if self.attention_mode == "can" and self.classifier_mode != "prototype":
            raise ValueError("attention_mode='can' requires classifier_mode='prototype'")
        if self.attention_mode == "can" and self.n_way < 2:
            raise ValueError("attention_mode='can' requires at least two task classes")
        self.can_attention_temperature = float(self.can_attention_temperature)
        if self.can_attention_temperature <= 0:
            raise ValueError("can_attention_temperature must be > 0")
        self.can_meta_hidden_dim = int(self.can_meta_hidden_dim)
        if self.can_meta_hidden_dim <= 0:
            raise ValueError("can_meta_hidden_dim must be > 0")
        self.can_local_loss_weight = float(self.can_local_loss_weight)
        if self.can_local_loss_weight < 0:
            raise ValueError("can_local_loss_weight must be non-negative")
        self.can_global_loss_weight = float(self.can_global_loss_weight)
        if self.can_global_loss_weight < 0:
            raise ValueError("can_global_loss_weight must be non-negative")
        self.can_transductive_iterations = int(self.can_transductive_iterations)
        if self.can_transductive_iterations < 0:
            raise ValueError("can_transductive_iterations must be non-negative")
        self.can_transductive_top_k_per_class = int(
            self.can_transductive_top_k_per_class
        )
        if self.can_transductive_top_k_per_class <= 0:
            raise ValueError("can_transductive_top_k_per_class must be > 0")
        self.can_transductive_min_confidence = float(
            self.can_transductive_min_confidence
        )
        if self.triplet_loss_weight < 0:
            raise ValueError("triplet_loss_weight must be non-negative")
        if self.triplet_margin < 0:
            raise ValueError("triplet_margin must be non-negative")
        if self.triplet_mining_strategy not in {
            "batch_hard",
            "batch_all",
            "triplet_center",
        }:
            raise ValueError(
                "triplet_mining_strategy must be one of: "
                "'batch_hard', 'batch_all', 'triplet_center'"
            )
        if self.triplet_center_gradient_clip_norm < 0:
            raise ValueError("triplet_center_gradient_clip_norm must be non-negative")
        self.embedding_batch_size = int(self.embedding_batch_size)
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be > 0")
        if self.num_sensors <= 0:
            raise ValueError("num_sensors must be > 0")
        if self.eegnet_temporal_filters <= 0:
            raise ValueError("eegnet_temporal_filters must be > 0")
        if self.eegnet_depth_multiplier <= 0:
            raise ValueError("eegnet_depth_multiplier must be > 0")
        if self.eegnet_separable_filters <= 0:
            raise ValueError("eegnet_separable_filters must be > 0")
        if self.eegnet_temporal_kernel_size <= 0:
            raise ValueError("eegnet_temporal_kernel_size must be > 0")
        if self.eegnet_separable_kernel_size <= 0:
            raise ValueError("eegnet_separable_kernel_size must be > 0")
        if self.eegnet_pool_size_1 <= 0:
            raise ValueError("eegnet_pool_size_1 must be > 0")
        if self.eegnet_pool_size_2 <= 0:
            raise ValueError("eegnet_pool_size_2 must be > 0")
        if self.eegnet_dropout_rate < 0 or self.eegnet_dropout_rate >= 1:
            raise ValueError("eegnet_dropout_rate must be in [0, 1)")
        if self.eegnet_l2_weight < 0:
            raise ValueError("eegnet_l2_weight must be non-negative")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be > 0")
        if self.window_shift_window_seconds <= 0:
            raise ValueError("window_shift_window_seconds must be > 0")
        if self.window_shift_step_seconds <= 0:
            raise ValueError("window_shift_step_seconds must be > 0")
        if self.window_shift_start_min_seconds < 0:
            raise ValueError("window_shift_start_min_seconds must be >= 0")
        if self.window_shift_start_max_seconds < self.window_shift_start_min_seconds:
            raise ValueError(
                "window_shift_start_max_seconds must be >= window_shift_start_min_seconds"
            )
        if self.gaussian_noise_std < 0:
            raise ValueError("gaussian_noise_std must be non-negative")
        if self.logging_verbosity not in {0, 1, 2}:
            raise ValueError("logging_verbosity must be one of: 0, 1, 2")
        if self.train_prefetch_batches <= 0:
            raise ValueError("train_prefetch_batches must be > 0")
        if self.gradient_clip_norm is not None:
            self.gradient_clip_norm = float(self.gradient_clip_norm)
            if self.gradient_clip_norm <= 0:
                raise ValueError("gradient_clip_norm must be positive or None")
        if self.train_progress_write_every_n_batches <= 0:
            raise ValueError("train_progress_write_every_n_batches must be > 0")
        if self.csv_flush_every_events <= 0:
            raise ValueError("csv_flush_every_events must be > 0")
        self.validation_checkpoint_metric = str(
            self.validation_checkpoint_metric
        ).strip()
        if (
            self.validation_checkpoint_metric
            not in SUPPORTED_VALIDATION_CHECKPOINT_METRICS
        ):
            raise ValueError(
                "validation_checkpoint_metric must be one of: "
                + ", ".join(SUPPORTED_VALIDATION_CHECKPOINT_METRICS)
            )
        self.validation_checkpoint_mode = str(self.validation_checkpoint_mode).strip()
        if self.validation_checkpoint_mode not in VALIDATION_CHECKPOINT_MODES:
            raise ValueError(
                "validation_checkpoint_mode must be one of: "
                + ", ".join(VALIDATION_CHECKPOINT_MODES)
            )
        self.lr_schedule = str(self.lr_schedule).strip().lower()
        if self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError("lr_schedule must be one of: 'constant', 'cosine'")
        self.lr_decay_alpha = float(self.lr_decay_alpha)
        if self.lr_decay_alpha < 0 or self.lr_decay_alpha > 1:
            raise ValueError("lr_decay_alpha must be in [0, 1]")
        if self.loso_start_index is not None:
            self.loso_start_index = int(self.loso_start_index)
            if self.loso_start_index <= 0:
                raise ValueError("loso_start_index must be a positive 1-based index")
        if self.loso_stop_index is not None:
            self.loso_stop_index = int(self.loso_stop_index)
            if self.loso_stop_index <= 0:
                raise ValueError("loso_stop_index must be a positive 1-based index")
        if (
            self.loso_start_index is not None
            and self.loso_stop_index is not None
            and self.loso_stop_index < self.loso_start_index
        ):
            raise ValueError("loso_stop_index must be >= loso_start_index")
        if self.data_variant not in {"real", "mock"}:
            raise ValueError("data_variant must be one of: 'real', 'mock'")
        if self.data_variant == "mock":
            self.data_path = "X_pre_mock.npy"
            self.labels_path = "y_heater_mock.npy"
            self.subjects_path = "subjects_mock.npy"
