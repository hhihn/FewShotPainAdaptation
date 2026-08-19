"""Early- and late-fusion CNNs plus their bounded search specifications."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import tensorflow as tf
from tensorflow import keras


BASE_FILTER_SCHEDULE = (16, 32, 32, 64, 64)
LATE_FILTER_SCHEDULE = (16, 16, 32, 32, 64, 64, 128)
LATE_OUTPUT_NAMES = ("eda_class", "emg_class", "ecg_class", "pain_class")
LATE_LOSS_WEIGHTS = {"eda_class": 0.2, "emg_class": 0.2, "ecg_class": 0.2, "pain_class": 0.4}


def _validate_dense_units(dense_units: tuple[int, ...]) -> None:
    if len(dense_units) not in {1, 2}:
        raise ValueError("The classifier must contain one or two hidden dense layers")
    if len(dense_units) == 2 and dense_units[1] > dense_units[0]:
        raise ValueError("Dense widths must be non-increasing")


@dataclass(frozen=True)
class ArchitectureSpec:
    """The original, searchable early-fusion architecture."""

    num_blocks: int
    conv_repeats: tuple[int, ...]
    width_multiplier: float
    temporal_kernel_size: int
    dense_units: tuple[int, ...]
    learning_rate: float
    dropout_rate: float = 0.25
    head_type: str = "flatten"
    convolution_type: str = "standard"
    normalization_type: str = "batch"
    pooling_type: str = "max"
    pooling_size: int = 2
    fusion_mode: str = "early"

    def __post_init__(self) -> None:
        if self.fusion_mode != "early":
            raise ValueError("ArchitectureSpec is only valid for early fusion")
        if not 3 <= self.num_blocks <= 5 or len(self.conv_repeats) != self.num_blocks:
            raise ValueError("Early fusion requires 3–5 repeat-defined blocks")
        if any(value not in {1, 2} for value in self.conv_repeats):
            raise ValueError("Every convolutional block must contain one or two convolutions")
        if self.width_multiplier not in {0.5, 1.0, 2.0} or self.temporal_kernel_size not in {7, 11, 15}:
            raise ValueError("Unsupported early-fusion width multiplier or temporal kernel")
        _validate_dense_units(self.dense_units)
        if self.learning_rate <= 0 or self.dropout_rate != 0.25:
            raise ValueError("learning_rate must be positive and dropout must be 0.25")
        if self.head_type not in {"flatten", "global_average"} or self.convolution_type not in {"standard", "separable"}:
            raise ValueError("Unsupported classifier head or convolution type")
        if self.normalization_type not in {"batch", "group", "layer"} or self.pooling_type not in {"max", "average"} or self.pooling_size not in {2, 4}:
            raise ValueError("Unsupported normalization or pooling setting")

    @classmethod
    def baseline(cls) -> "ArchitectureSpec":
        return cls(5, (2, 1, 1, 1, 1), 1.0, 11, (1024, 512), 1e-5)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArchitectureSpec":
        return cls(
            num_blocks=int(payload["num_blocks"]), conv_repeats=tuple(int(value) for value in payload["conv_repeats"]),
            width_multiplier=float(payload["width_multiplier"]), temporal_kernel_size=int(payload["temporal_kernel_size"]),
            dense_units=tuple(int(value) for value in payload["dense_units"]), learning_rate=float(payload["learning_rate"]),
            dropout_rate=float(payload.get("dropout_rate", 0.25)), head_type=str(payload.get("head_type", "flatten")),
            convolution_type=str(payload.get("convolution_type", "standard")), normalization_type=str(payload.get("normalization_type", "batch")),
            pooling_type=str(payload.get("pooling_type", "max")), pooling_size=int(payload.get("pooling_size", 2)),
            fusion_mode=str(payload.get("fusion_mode", "early")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def block_filters(self) -> tuple[int, ...]:
        return tuple(max(1, int(round(f * self.width_multiplier))) for f in BASE_FILTER_SCHEDULE[: self.num_blocks])


@dataclass(frozen=True)
class LateBranchSpec:
    num_blocks: int
    conv_repeats: tuple[int, ...]
    width_multiplier: float
    temporal_kernel_size: int
    dense_units: tuple[int, ...]
    head_type: str = "flatten"

    def __post_init__(self) -> None:
        if not 5 <= self.num_blocks <= 7 or len(self.conv_repeats) != self.num_blocks:
            raise ValueError("Late-fusion branches require 5–7 repeat-defined blocks")
        if any(value not in {1, 2} for value in self.conv_repeats) or self.width_multiplier not in {0.5, 1.0, 2.0}:
            raise ValueError("Unsupported late-fusion convolution setting")
        if self.temporal_kernel_size <= 0 or self.head_type not in {"flatten", "global_average"}:
            raise ValueError("Unsupported late-fusion kernel or head")
        _validate_dense_units(self.dense_units)

    @classmethod
    def reference(cls, kernel: int) -> "LateBranchSpec":
        return cls(7, (1,) * 7, 1.0, kernel, (1024, 512))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LateBranchSpec":
        return cls(int(payload["num_blocks"]), tuple(int(v) for v in payload["conv_repeats"]), float(payload["width_multiplier"]), int(payload["temporal_kernel_size"]), tuple(int(v) for v in payload["dense_units"]), str(payload.get("head_type", "flatten")))

    def block_filters(self) -> tuple[int, ...]:
        return tuple(max(1, int(round(f * self.width_multiplier))) for f in LATE_FILTER_SCHEDULE[: self.num_blocks])


@dataclass(frozen=True)
class LateFusionArchitectureSpec:
    eda: LateBranchSpec
    emg: LateBranchSpec
    ecg: LateBranchSpec
    learning_rate: float
    dropout_rate: float = 0.25
    fusion_mode: str = "late"

    def __post_init__(self) -> None:
        if self.fusion_mode != "late" or self.learning_rate <= 0 or self.dropout_rate != 0.25:
            raise ValueError("Late fusion requires a positive learning rate and 0.25 dropout")

    @classmethod
    def baseline(cls) -> "LateFusionArchitectureSpec":
        return cls(LateBranchSpec.reference(3), LateBranchSpec.reference(11), LateBranchSpec.reference(11), 1e-5)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LateFusionArchitectureSpec":
        return cls(LateBranchSpec.from_dict(payload["eda"]), LateBranchSpec.from_dict(payload["emg"]), LateBranchSpec.from_dict(payload["ecg"]), float(payload["learning_rate"]), float(payload.get("dropout_rate", 0.25)), str(payload.get("fusion_mode", "late")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ModelSpec = ArchitectureSpec | LateFusionArchitectureSpec


def architecture_from_dict(payload: Mapping[str, Any]) -> ModelSpec:
    return LateFusionArchitectureSpec.from_dict(payload) if payload.get("fusion_mode") == "late" else ArchitectureSpec.from_dict(payload)


def build_early_fusion_model(spec: ArchitectureSpec, *, input_shape=(3, 1152, 1), num_classes=2) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="physiological_modalities")
    x = inputs
    for block_index, (filters, repeat_count) in enumerate(zip(spec.block_filters(), spec.conv_repeats), 1):
        convolution_class = keras.layers.Conv2D if spec.convolution_type == "standard" else keras.layers.SeparableConv2D
        kernel_height = 2 if block_index == 1 else 1
        for repeat_index in range(1, repeat_count + 1):
            x = convolution_class(filters, (kernel_height, spec.temporal_kernel_size), padding="same", activation="elu", name=f"block_{block_index}_{'conv' if spec.convolution_type == 'standard' else 'separable_conv'}_{repeat_index}")(x)
        if spec.normalization_type == "batch":
            x = keras.layers.BatchNormalization(name=f"block_{block_index}_batch_norm")(x)
        elif spec.normalization_type == "group":
            x = keras.layers.GroupNormalization(groups=min(8, filters), axis=-1, name=f"block_{block_index}_group_norm")(x)
        else:
            x = keras.layers.LayerNormalization(axis=-1, name=f"block_{block_index}_layer_norm")(x)
        pool = keras.layers.MaxPooling2D if spec.pooling_type == "max" else keras.layers.AveragePooling2D
        x = pool(pool_size=(1, spec.pooling_size), strides=(1, spec.pooling_size), name=f"block_{block_index}_{spec.pooling_type}_pool")(x)
        x = keras.layers.Dropout(spec.dropout_rate, name=f"block_{block_index}_dropout")(x)
    x = keras.layers.Flatten(name="flatten")(x) if spec.head_type == "flatten" else keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = keras.layers.Dropout(spec.dropout_rate, name="head_dropout")(x)
    for index, units in enumerate(spec.dense_units, 1):
        x = keras.layers.Dense(units, activation="elu", name=f"dense_{index}")(x)
        x = keras.layers.Dropout(spec.dropout_rate, name=f"dense_{index}_dropout")(x)
    return keras.Model(inputs, keras.layers.Dense(num_classes, activation="softmax", name="pain_class")(x), name="painnas_early_fusion")


class NormalizedLateFusion(keras.layers.Layer):
    """Non-negative, normalized modality weights followed by a linear sum."""

    def build(self, input_shape) -> None:
        self.alpha = self.add_weight(name="alpha", shape=(3,), initializer=keras.initializers.Constant(1 / 3), constraint=keras.constraints.NonNeg(), trainable=True)

    def call(self, inputs):
        positive = self.alpha + keras.backend.epsilon()
        weights = positive / tf.reduce_sum(positive)
        return tf.add_n([weight * value for weight, value in zip(tf.unstack(weights), inputs)])


def _branch(inputs, spec: LateBranchSpec, *, name: str, dropout_rate: float, post_pool_dropout: bool, vector_dropout: bool, num_classes: int):
    x = inputs
    for block_index, (filters, repeat_count) in enumerate(zip(spec.block_filters(), spec.conv_repeats), 1):
        for repeat_index in range(1, repeat_count + 1):
            x = keras.layers.Conv1D(filters, spec.temporal_kernel_size, padding="same", activation="elu", name=f"{name}_block_{block_index}_conv_{repeat_index}")(x)
        x = keras.layers.BatchNormalization(name=f"{name}_block_{block_index}_batch_norm")(x)
        x = keras.layers.MaxPooling1D(2, strides=2, name=f"{name}_block_{block_index}_max_pool")(x)
        if post_pool_dropout:
            x = keras.layers.Dropout(dropout_rate, name=f"{name}_block_{block_index}_dropout")(x)
    x = keras.layers.Flatten(name=f"{name}_flatten")(x) if spec.head_type == "flatten" else keras.layers.GlobalAveragePooling1D(name=f"{name}_global_average_pooling")(x)
    if vector_dropout:
        x = keras.layers.Dropout(dropout_rate, name=f"{name}_head_dropout")(x)
    for index, units in enumerate(spec.dense_units, 1):
        x = keras.layers.Dense(units, activation="elu", name=f"{name}_dense_{index}")(x)
        x = keras.layers.Dropout(dropout_rate, name=f"{name}_dense_{index}_dropout")(x)
    return keras.layers.Dense(num_classes, activation="softmax", name=f"{name}_class")(x)


def _modality_indices(modalities: tuple[str, ...]) -> dict[str, int]:
    aliases = {"GSR": "EDA", "EDA": "EDA", "ECG": "ECG", "EMG": "EMG"}
    canonical = [aliases.get(name.upper(), name.upper()) for name in modalities]
    if len(canonical) != 3 or set(canonical) != {"EDA", "ECG", "EMG"}:
        raise ValueError("Late fusion requires exactly EDA/GSR, ECG, and EMG")
    return {name: canonical.index(name) for name in canonical}


def build_late_fusion_model(spec: LateFusionArchitectureSpec, *, input_shape=(3, 1152, 1), num_classes=2, modalities: tuple[str, ...] = ("GSR", "ECG", "EMG")) -> keras.Model:
    if input_shape[0] != 3:
        raise ValueError("Late fusion requires three modality channels")
    inputs = keras.Input(shape=input_shape, name="physiological_modalities")
    selected = {name: keras.layers.Lambda(lambda value, index=index: value[:, index, :, :], name=f"select_{name.lower()}")(inputs) for name, index in _modality_indices(modalities).items()}
    eda = _branch(selected["EDA"], spec.eda, name="eda", dropout_rate=spec.dropout_rate, post_pool_dropout=False, vector_dropout=False, num_classes=num_classes)
    emg = _branch(selected["EMG"], spec.emg, name="emg", dropout_rate=spec.dropout_rate, post_pool_dropout=True, vector_dropout=True, num_classes=num_classes)
    ecg = _branch(selected["ECG"], spec.ecg, name="ecg", dropout_rate=spec.dropout_rate, post_pool_dropout=True, vector_dropout=True, num_classes=num_classes)
    aggregate = NormalizedLateFusion(name="pain_class")([eda, emg, ecg])
    return keras.Model(inputs, [eda, emg, ecg, aggregate], name="painnas_late_fusion")


def build_model(spec: ModelSpec, *, input_shape=(3, 1152, 1), num_classes=2, modalities: tuple[str, ...] = ("GSR", "ECG", "EMG")) -> keras.Model:
    return build_late_fusion_model(spec, input_shape=input_shape, num_classes=num_classes, modalities=modalities) if isinstance(spec, LateFusionArchitectureSpec) else build_early_fusion_model(spec, input_shape=input_shape, num_classes=num_classes)


def target_output_names(spec: ModelSpec) -> tuple[str, ...]:
    return LATE_OUTPUT_NAMES if isinstance(spec, LateFusionArchitectureSpec) else ("pain_class",)


def validation_monitor(spec: ModelSpec) -> str:
    return "val_pain_class_macro_f1" if isinstance(spec, LateFusionArchitectureSpec) else "val_macro_f1"


def aggregate_probabilities(model: keras.Model, prediction: Any) -> Any:
    if isinstance(prediction, Mapping):
        return prediction["pain_class"]
    if isinstance(prediction, (list, tuple)):
        return prediction[list(model.output_names).index("pain_class")]
    return prediction


def learned_fusion_weights(model: keras.Model) -> dict[str, float] | None:
    try:
        layer = model.get_layer("pain_class")
    except ValueError:
        return None
    if not isinstance(layer, NormalizedLateFusion):
        return None
    values = layer.alpha.numpy() + keras.backend.epsilon()
    values = values / values.sum()
    return {name: float(value) for name, value in zip(("EDA", "EMG", "ECG"), values)}


def compile_model(model: keras.Model, spec: ModelSpec) -> keras.Model:
    optimizer = keras.optimizers.Adam(learning_rate=spec.learning_rate)
    metrics = [keras.metrics.CategoricalAccuracy(name="accuracy"), keras.metrics.F1Score(name="macro_f1", average="macro")]
    if isinstance(spec, LateFusionArchitectureSpec):
        model.compile(optimizer=optimizer, loss={name: keras.losses.CategoricalCrossentropy() for name in LATE_OUTPUT_NAMES}, loss_weights=LATE_LOSS_WEIGHTS, metrics={"pain_class": metrics})
    else:
        model.compile(optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(), metrics=metrics)
    return model


def early_stopping_callbacks(*, monitor: str, patience: int) -> list[keras.callbacks.Callback]:
    return [keras.callbacks.TerminateOnNaN(), keras.callbacks.EarlyStopping(monitor=monitor, mode="max", patience=int(patience), restore_best_weights=True)]
