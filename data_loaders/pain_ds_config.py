from typing import Tuple, Optional
from dataclasses import dataclass


SUPPORTED_VALIDATION_CHECKPOINT_METRICS = (
    "loss",
    "task_loss",
    "can_local_loss",
    "can_margin_loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "can_true_class_score",
    "can_best_other_score",
    "can_score_margin",
)

VALIDATION_CHECKPOINT_MODES = ("auto", "min", "max")
CAN_SUPPORT_MODES = ("sampled", "learned_prototype_memory")
PROTOTYPE_PHASE2_LOSS_MODES = ("ce_can",)
SOURCE_SUBJECT_PROTOTYPE_VOTE_SOFTMAX_SCOPES = ("global",)
SUPPORTED_DATASET_SOURCES = ("painmonit", "biovid_part_a", "senseemotion")
PREDEFINED_SPLIT_DATASET_SOURCES = ("biovid_part_a", "senseemotion")
ALLOWED_MODALITIES_BY_DATASET = {
    "biovid_part_a": ("ECG", "EMG", "GSR"),
    "senseemotion": ("ECG", "EMG", "GSR", "RSP"),
}
MODALITY_ALIASES = {"EDA": "GSR", "GRS": "GSR"}


@dataclass
class PainDatasetConfig:
    """Store dataset, episodic sampling, and model hyperparameters.

    The dataclass normalizes coupled settings in ``__post_init__`` so downstream
    loaders and learners can rely on a consistent configuration.
    """

    # Data dimensions
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
    eegnet_normalization: str = "group"
    eegnet_group_norm_groups: int = 4
    encoder_backend: str = "eegnet"
    crossmod_num_heads: int = 8
    crossmod_hidden_dim: int = 128
    crossmod_num_layers: int = 2
    crossmod_positional_base: float = 10000.0
    crossmod_attention_dropout_rate: float = 0.0
    crossmod_ff_activation: str = "relu"
    clear_session_per_fold: bool = True  # Legacy flag; LOSO folds now reuse one graph
    single_loso_fold: bool = True  # If True, run only one LOSO fold (testing mode)
    single_loso_test_subject: Optional[int] = None  # Optional explicit held-out subject
    loso_start_index: Optional[int] = None  # 1-based inclusive LOSO fold start
    loso_stop_index: Optional[int] = None  # 1-based inclusive LOSO fold stop
    # Modality information
    modality_names: Tuple[str, ...] = (
        "EDA",  # idx 1
        "ECG",  # idx 4
        "EMG",  # idx 5
    )
    # Sensor to index mapping
    sensor_idx: Tuple[int, ...] = (1, 4, 5)

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
    attention_mode: str = "can"  # CAN over temporal feature maps
    can_attention_temperature: float = 1.0
    can_meta_hidden_dim: int = 32
    can_local_pool_temperature: float = 0.1
    can_local_loss_weight: float = 1.0
    can_margin_loss_weight: float = 0.2
    can_margin_target: float = 0.3
    can_support_mode: str = "sampled"
    learned_prototype_slots_per_class: int = 1
    prototype_bank_init_samples_per_class: int = 0
    prototype_finetune_epochs: int = 1
    prototype_finetune_tasks_per_epoch: Optional[int] = None
    prototype_phase2_loss_mode: str = "ce_can"
    source_subject_prototype_vote_enabled: bool = True
    source_subject_prototype_vote_use_base_index: bool = True
    source_subject_prototype_vote_query_normalize_with_subject_stats: bool = True
    source_subject_prototype_vote_softmax_scope: str = "global"
    train_batch_size: int = 256  # Number of tasks per optimizer update
    task_chunk_size: int = 1  # Number of tasks encoded together per forward chunk
    num_epochs: int = 10  # Number of epochs per fold
    tasks_per_epoch: int = 100  # Number of train tasks sampled per epoch
    val_tasks: int = 20  # Number of validation tasks per validation run
    heldout_eval_tasks: int = 20  # Number of held-out evaluation tasks per fold
    subject_eval_tasks: Optional[int] = None  # Deprecated alias for heldout_eval_tasks
    matched_query_eval: bool = False
    matched_query_support_repeats: int = 500
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
    disable_validation: bool = False  # Skip validation/checkpoint evaluation entirely
    logging_verbosity: int = 1  # 0=minimal, 1=standard, 2=detailed training logs
    disable_training_logging: bool = False  # Skip train progress logs and CSV updates
    train_prefetch_batches: int = 256  # Number of asynchronously prepared train batches
    gradient_clip_norm: Optional[float] = (
        1.0  # Per-gradient norm clip for optimizer updates
    )
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
    dataset_source: str = "biovid_part_a"  # painmonit, biovid_part_a, or senseemotion
    split_strategy: str = "loso"  # loso or predefined
    data_variant: str = "real"  # real or mock
    data_path: str = "X_pre.npy"
    labels_path: str = "y_heater.npy"
    subjects_path: str = "subjects.npy"
    biovid_part_dir: str = "PartA"
    biovid_train_split_dir: str = "Train"
    biovid_test_split_dir: str = "Test"
    biovid_modalities: Tuple[str, ...] = ("GSR", "ECG", "EMG")
    senseemotion_dir: str = "SenseEmotion"
    senseemotion_train_split_dir: str = "Train"
    senseemotion_test_split_dir: str = "Test"
    senseemotion_modalities: Tuple[str, ...] = ("GSR", "ECG")
    modalities: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        """Normalize derived fields and validate configuration values.

        Raises:
            ValueError: If dataset, model, sampler, or training settings are
                inconsistent or outside supported ranges.
        """
        self.dataset_source = str(self.dataset_source).strip().lower()
        if self.dataset_source not in SUPPORTED_DATASET_SOURCES:
            raise ValueError(
                "dataset_source must be one of: " + ", ".join(SUPPORTED_DATASET_SOURCES)
            )
        if self.subject_eval_tasks is not None:
            self.heldout_eval_tasks = int(self.subject_eval_tasks)
        self.matched_query_support_repeats = int(self.matched_query_support_repeats)
        if self.matched_query_support_repeats < 1:
            raise ValueError("matched_query_support_repeats must be >= 1")
        self.encoder_backend = str(self.encoder_backend).strip().lower()
        if self.encoder_backend not in {"eegnet", "crossmod"}:
            raise ValueError("encoder_backend must be one of: 'eegnet', 'crossmod'")
        if self.encoder_backend == "crossmod":
            if str(self.attention_mode).strip().lower() != "can":
                raise ValueError(
                    "encoder_backend='crossmod' requires attention_mode='can'"
                )
            if self.dataset_source == "painmonit":
                self.num_sensors = 2
                self.sensor_idx = (1, 4)
                self.modality_names = ("EDA", "ECG")
        if self.split_strategy not in {"loso", "predefined"}:
            raise ValueError("split_strategy must be one of: 'loso', 'predefined'")
        if self.dataset_source in PREDEFINED_SPLIT_DATASET_SOURCES:
            configured_modalities = self.modalities
            if configured_modalities is None:
                if self.encoder_backend == "crossmod":
                    configured_modalities = ("GSR", "ECG")
                else:
                    configured_modalities = (
                        self.biovid_modalities
                        if self.dataset_source == "biovid_part_a"
                        else self.senseemotion_modalities
                    )
            if isinstance(configured_modalities, str):
                configured_modalities = tuple(configured_modalities.split(","))
            normalized_modalities = tuple(
                MODALITY_ALIASES.get(
                    str(modality).strip().upper(), str(modality).strip().upper()
                )
                for modality in configured_modalities
            )
            if (
                self.modalities is not None or self.encoder_backend == "crossmod"
            ) and len(normalized_modalities) != 2:
                raise ValueError("modalities must contain exactly two modalities")
            if len(set(normalized_modalities)) != len(normalized_modalities):
                raise ValueError(
                    "modalities must contain distinct modalities; GSR and EDA are synonyms"
                )
            allowed_modalities = ALLOWED_MODALITIES_BY_DATASET[self.dataset_source]
            invalid_modalities = tuple(
                modality
                for modality in normalized_modalities
                if modality not in allowed_modalities
            )
            if invalid_modalities:
                raise ValueError(
                    f"Unsupported modalities for {self.dataset_source}: "
                    f"{', '.join(invalid_modalities)}. Allowed: "
                    f"{', '.join(allowed_modalities)} (EDA is an alias for GSR)"
                )
            self.modalities = normalized_modalities
            self.num_sensors = len(normalized_modalities)
            self.sensor_idx = tuple(range(self.num_sensors))
            self.modality_names = tuple(
                "EDA" if modality == "GSR" else modality
                for modality in normalized_modalities
            )
            if self.dataset_source == "biovid_part_a":
                self.biovid_modalities = normalized_modalities
            else:
                self.senseemotion_modalities = normalized_modalities
            # BioVid Part A and SenseEmotion ship with explicit train/test splits.
            self.split_strategy = "predefined"
            # Avoid applying an additional sliding-window augmentation on top of
            # pre-segmented windows.
            self.enable_window_shift_augmentation = False
            if self.data_variant == "mock":
                raise ValueError(
                    "data_variant='mock' is only supported for dataset_source='painmonit'"
                )

        if self.dataset_source == "biovid_part_a":
            self.sampling_rate_hz = 256
            if self.task_class_ids == (0, 5):
                self.task_class_ids = (0, 4)
        elif self.dataset_source == "senseemotion":
            self.sequence_length = 1664
            if self.task_class_ids == (0, 5):
                self.task_class_ids = (0, 1, 2, 3)

        if self.matched_query_eval:
            if (
                self.dataset_source not in PREDEFINED_SPLIT_DATASET_SOURCES
                or self.split_strategy != "predefined"
            ):
                raise ValueError(
                    "matched_query_eval requires a predefined Train/Test split dataset"
                )
            if self.task_normalize_mode != "split":
                raise ValueError(
                    "matched_query_eval requires task_normalize_mode='split' so "
                    "both conditions use frozen source-only statistics"
                )
            if self.can_support_mode != "learned_prototype_memory":
                raise ValueError(
                    "matched_query_eval requires can_support_mode='learned_prototype_memory'"
                )
            if int(self.k_shot_adaptation_steps) != 0:
                raise ValueError(
                    "matched_query_eval requires k_shot_adaptation_steps=0; the "
                    "tested personalization operation is sampled support prototypes"
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
        self.attention_mode = str(self.attention_mode).strip().lower()
        if self.attention_mode != "can":
            raise ValueError("attention_mode must be 'can'")
        if self.n_way < 2:
            raise ValueError("attention_mode='can' requires at least two task classes")
        self.can_attention_temperature = float(self.can_attention_temperature)
        if self.can_attention_temperature <= 0:
            raise ValueError("can_attention_temperature must be > 0")
        self.can_meta_hidden_dim = int(self.can_meta_hidden_dim)
        if self.can_meta_hidden_dim <= 0:
            raise ValueError("can_meta_hidden_dim must be > 0")
        self.can_local_pool_temperature = float(self.can_local_pool_temperature)
        if self.can_local_pool_temperature <= 0:
            raise ValueError("can_local_pool_temperature must be > 0")
        self.can_local_loss_weight = float(self.can_local_loss_weight)
        if self.can_local_loss_weight < 0:
            raise ValueError("can_local_loss_weight must be non-negative")
        self.can_margin_loss_weight = float(self.can_margin_loss_weight)
        if self.can_margin_loss_weight < 0:
            raise ValueError("can_margin_loss_weight must be non-negative")
        self.can_margin_target = float(self.can_margin_target)
        if self.can_margin_target < 0:
            raise ValueError("can_margin_target must be non-negative")
        self.can_support_mode = str(self.can_support_mode).strip().lower()
        if self.can_support_mode not in CAN_SUPPORT_MODES:
            raise ValueError(
                "can_support_mode must be one of: " + ", ".join(CAN_SUPPORT_MODES)
            )
        self.learned_prototype_slots_per_class = int(
            self.learned_prototype_slots_per_class
        )
        if self.learned_prototype_slots_per_class <= 0:
            raise ValueError("learned_prototype_slots_per_class must be > 0")
        self.prototype_bank_init_samples_per_class = int(
            self.prototype_bank_init_samples_per_class
        )
        if self.prototype_bank_init_samples_per_class < 0:
            raise ValueError("prototype_bank_init_samples_per_class must be >= 0")
        self.prototype_finetune_epochs = int(self.prototype_finetune_epochs)
        if self.prototype_finetune_epochs < 0:
            raise ValueError("prototype_finetune_epochs must be non-negative")
        if self.prototype_finetune_tasks_per_epoch is not None:
            self.prototype_finetune_tasks_per_epoch = int(
                self.prototype_finetune_tasks_per_epoch
            )
            if self.prototype_finetune_tasks_per_epoch <= 0:
                raise ValueError("prototype_finetune_tasks_per_epoch must be > 0")
        self.prototype_phase2_loss_mode = (
            str(self.prototype_phase2_loss_mode).strip().lower()
        )
        if self.prototype_phase2_loss_mode not in PROTOTYPE_PHASE2_LOSS_MODES:
            raise ValueError(
                "prototype_phase2_loss_mode must be one of: "
                + ", ".join(PROTOTYPE_PHASE2_LOSS_MODES)
            )
        self.source_subject_prototype_vote_enabled = bool(
            self.source_subject_prototype_vote_enabled
        )
        self.source_subject_prototype_vote_use_base_index = bool(
            self.source_subject_prototype_vote_use_base_index
        )
        self.source_subject_prototype_vote_query_normalize_with_subject_stats = bool(
            self.source_subject_prototype_vote_query_normalize_with_subject_stats
        )
        self.source_subject_prototype_vote_softmax_scope = (
            str(self.source_subject_prototype_vote_softmax_scope).strip().lower()
        )
        if (
            self.source_subject_prototype_vote_softmax_scope
            not in SOURCE_SUBJECT_PROTOTYPE_VOTE_SOFTMAX_SCOPES
        ):
            raise ValueError(
                "source_subject_prototype_vote_softmax_scope must be one of: "
                + ", ".join(SOURCE_SUBJECT_PROTOTYPE_VOTE_SOFTMAX_SCOPES)
            )
        if self.can_support_mode == "learned_prototype_memory":
            if self.attention_mode != "can":
                raise ValueError(
                    "can_support_mode='learned_prototype_memory' requires attention_mode='can'"
                )
        elif self.prototype_bank_init_samples_per_class > 0:
            raise ValueError(
                "prototype_bank_init_samples_per_class > 0 requires "
                "can_support_mode='learned_prototype_memory'"
            )
        self.task_chunk_size = int(self.task_chunk_size)
        if self.task_chunk_size <= 0:
            raise ValueError("task_chunk_size must be > 0")
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
        self.eegnet_normalization = str(self.eegnet_normalization).strip().lower()
        if self.eegnet_normalization not in {"group", "layer"}:
            raise ValueError("eegnet_normalization must be one of: group, layer")
        self.eegnet_group_norm_groups = int(self.eegnet_group_norm_groups)
        if self.eegnet_group_norm_groups <= 0:
            raise ValueError("eegnet_group_norm_groups must be > 0")
        crossmod_positive_ints = {
            "crossmod_num_heads": self.crossmod_num_heads,
            "crossmod_hidden_dim": self.crossmod_hidden_dim,
            "crossmod_num_layers": self.crossmod_num_layers,
        }
        for field_name, field_value in crossmod_positive_ints.items():
            if int(field_value) <= 0:
                raise ValueError(f"{field_name} must be > 0")
        self.crossmod_attention_dropout_rate = float(
            self.crossmod_attention_dropout_rate
        )
        if (
            self.crossmod_attention_dropout_rate < 0
            or self.crossmod_attention_dropout_rate >= 1
        ):
            raise ValueError("crossmod_attention_dropout_rate must be in [0, 1)")
        self.crossmod_positional_base = float(self.crossmod_positional_base)
        if self.crossmod_positional_base <= 0:
            raise ValueError("crossmod_positional_base must be > 0")
        if (
            self.encoder_backend == "crossmod"
            and int(self.eegnet_separable_filters) % int(self.crossmod_num_heads) != 0
        ):
            raise ValueError(
                "eegnet_separable_filters must be divisible by crossmod_num_heads"
            )
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
