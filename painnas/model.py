"""Table-2 early-fusion CNN and its bounded architecture search space."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from tensorflow import keras


BASE_FILTER_SCHEDULE = (16, 32, 32, 64, 64)


@dataclass(frozen=True)
class ArchitectureSpec:
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

    def __post_init__(self) -> None:
        if not 3 <= self.num_blocks <= 5:
            raise ValueError("num_blocks must be between 3 and 5")
        if len(self.conv_repeats) != self.num_blocks:
            raise ValueError("conv_repeats must contain one value per block")
        if any(value not in {1, 2} for value in self.conv_repeats):
            raise ValueError("Every convolutional block must contain one or two convolutions")
        if self.width_multiplier not in {0.5, 1.0, 2.0}:
            raise ValueError("Unsupported width multiplier")
        if self.temporal_kernel_size not in {7, 11, 15}:
            raise ValueError("Unsupported temporal kernel size")
        if len(self.dense_units) not in {1, 2}:
            raise ValueError("The classifier must contain one or two hidden dense layers")
        if len(self.dense_units) == 2 and self.dense_units[1] > self.dense_units[0]:
            raise ValueError("Dense widths must be non-increasing")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.dropout_rate != 0.25:
            raise ValueError("The requested architecture fixes dropout at 0.25")
        if self.head_type not in {"flatten", "global_average"}:
            raise ValueError("Unsupported classifier head type")
        if self.convolution_type not in {"standard", "separable"}:
            raise ValueError("Unsupported convolution type")
        if self.normalization_type not in {"batch", "group", "layer"}:
            raise ValueError("Unsupported normalization type")
        if self.pooling_type not in {"max", "average"}:
            raise ValueError("Unsupported pooling type")
        if self.pooling_size not in {2, 4}:
            raise ValueError("pooling_size must be 2 or 4")

    @classmethod
    def baseline(cls) -> "ArchitectureSpec":
        return cls(
            num_blocks=5,
            conv_repeats=(2, 1, 1, 1, 1),
            width_multiplier=1.0,
            temporal_kernel_size=11,
            dense_units=(1024, 512),
            learning_rate=1e-5,
            head_type="flatten",
            convolution_type="standard",
            normalization_type="batch",
            pooling_type="max",
            pooling_size=2,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArchitectureSpec":
        return cls(
            num_blocks=int(payload["num_blocks"]),
            conv_repeats=tuple(int(value) for value in payload["conv_repeats"]),
            width_multiplier=float(payload["width_multiplier"]),
            temporal_kernel_size=int(payload["temporal_kernel_size"]),
            dense_units=tuple(int(value) for value in payload["dense_units"]),
            learning_rate=float(payload["learning_rate"]),
            dropout_rate=float(payload.get("dropout_rate", 0.25)),
            head_type=str(payload.get("head_type", "flatten")),
            convolution_type=str(payload.get("convolution_type", "standard")),
            normalization_type=str(payload.get("normalization_type", "batch")),
            pooling_type=str(payload.get("pooling_type", "max")),
            pooling_size=int(payload.get("pooling_size", 2)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def block_filters(self) -> tuple[int, ...]:
        return tuple(
            max(1, int(round(filters * self.width_multiplier)))
            for filters in BASE_FILTER_SCHEDULE[: self.num_blocks]
        )


def build_early_fusion_model(
    spec: ArchitectureSpec,
    *,
    input_shape: tuple[int, int, int] = (3, 1152, 1),
    num_classes: int = 2,
) -> keras.Model:
    """Build a fresh early-fusion model from an architecture specification."""

    inputs = keras.Input(shape=input_shape, name="physiological_modalities")
    x = inputs
    for block_index, (filters, repeat_count) in enumerate(
        zip(spec.block_filters(), spec.conv_repeats), start=1
    ):
        kernel_height = 2 if block_index == 1 else 1
        for repeat_index in range(1, repeat_count + 1):
            convolution_class = (
                keras.layers.Conv2D
                if spec.convolution_type == "standard"
                else keras.layers.SeparableConv2D
            )
            convolution_name = (
                f"block_{block_index}_conv_{repeat_index}"
                if spec.convolution_type == "standard"
                else f"block_{block_index}_separable_conv_{repeat_index}"
            )
            x = convolution_class(
                filters=filters,
                kernel_size=(kernel_height, spec.temporal_kernel_size),
                strides=(1, 1),
                padding="same",
                activation="elu",
                name=convolution_name,
            )(x)
        if spec.normalization_type == "batch":
            x = keras.layers.BatchNormalization(
                name=f"block_{block_index}_batch_norm"
            )(x)
        elif spec.normalization_type == "group":
            x = keras.layers.GroupNormalization(
                groups=min(8, filters),
                axis=-1,
                name=f"block_{block_index}_group_norm",
            )(x)
        else:
            x = keras.layers.LayerNormalization(
                axis=-1, name=f"block_{block_index}_layer_norm"
            )(x)
        pooling_class = (
            keras.layers.MaxPooling2D
            if spec.pooling_type == "max"
            else keras.layers.AveragePooling2D
        )
        x = pooling_class(
            pool_size=(1, spec.pooling_size),
            strides=(1, spec.pooling_size),
            name=f"block_{block_index}_{spec.pooling_type}_pool",
        )(x)
        x = keras.layers.Dropout(
            spec.dropout_rate, name=f"block_{block_index}_dropout"
        )(x)

    if spec.head_type == "flatten":
        x = keras.layers.Flatten(name="flatten")(x)
    else:
        x = keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = keras.layers.Dropout(spec.dropout_rate, name="head_dropout")(x)
    for dense_index, units in enumerate(spec.dense_units, start=1):
        x = keras.layers.Dense(
            units, activation="elu", name=f"dense_{dense_index}"
        )(x)
        x = keras.layers.Dropout(
            spec.dropout_rate, name=f"dense_{dense_index}_dropout"
        )(x)
    outputs = keras.layers.Dense(
        num_classes, activation="softmax", name="pain_class"
    )(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="painnas_early_fusion")


def compile_model(model: keras.Model, spec: ArchitectureSpec) -> keras.Model:
    optimizer = keras.optimizers.Adam(learning_rate=spec.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.F1Score(name="macro_f1", average="macro"),
        ],
    )
    return model


def early_stopping_callbacks(*, monitor: str, patience: int) -> list[keras.callbacks.Callback]:
    """Build the finite-loss and best-weight callbacks used by every fit."""

    return [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=int(patience),
            restore_best_weights=True,
        ),
    ]
