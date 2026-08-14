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

    @classmethod
    def baseline(cls) -> "ArchitectureSpec":
        return cls(
            num_blocks=5,
            conv_repeats=(2, 1, 1, 1, 1),
            width_multiplier=1.0,
            temporal_kernel_size=11,
            dense_units=(1024, 512),
            learning_rate=1e-5,
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
            x = keras.layers.Conv2D(
                filters=filters,
                kernel_size=(kernel_height, spec.temporal_kernel_size),
                strides=(1, 1),
                padding="same",
                activation="elu",
                name=f"block_{block_index}_conv_{repeat_index}",
            )(x)
        x = keras.layers.BatchNormalization(name=f"block_{block_index}_batch_norm")(x)
        x = keras.layers.MaxPooling2D(
            pool_size=(1, 2),
            strides=(1, 2),
            name=f"block_{block_index}_max_pool",
        )(x)
        x = keras.layers.Dropout(
            spec.dropout_rate, name=f"block_{block_index}_dropout"
        )(x)

    x = keras.layers.Flatten(name="flatten")(x)
    x = keras.layers.Dropout(spec.dropout_rate, name="flatten_dropout")(x)
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
