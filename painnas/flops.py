"""Analytical multiply-accumulate estimates for PainNAS models.

The estimates count convolution and dense-layer multiply-accumulates (MACs) for
one forward pass.  Reporting FLOPs as ``2 * MACs`` makes the convention
explicit: one multiplication and one addition are treated as two operations.
Bias addition, normalization, activation, pooling, dropout, softmax, and the
small late-fusion weighted sum are intentionally excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

from painnas.model import ArchitectureSpec, LateBranchSpec, ModelSpec


@dataclass(frozen=True)
class ComputeEstimate:
    """Per-sample forward-pass compute under the two-FLOPs-per-MAC convention."""

    convolution_macs: int
    dense_macs: int

    @property
    def macs(self) -> int:
        return self.convolution_macs + self.dense_macs

    @property
    def flops(self) -> int:
        return 2 * self.macs

    def __add__(self, other: "ComputeEstimate") -> "ComputeEstimate":
        if not isinstance(other, ComputeEstimate):
            return NotImplemented
        return ComputeEstimate(
            convolution_macs=self.convolution_macs + other.convolution_macs,
            dense_macs=self.dense_macs + other.dense_macs,
        )


def _dense_macs(input_width: int, hidden_units: tuple[int, ...], num_classes: int) -> int:
    macs = 0
    width = input_width
    for units in hidden_units:
        macs += width * units
        width = units
    return macs + width * num_classes


def estimate_early_fusion_compute(
    spec: ArchitectureSpec,
    *,
    sequence_length: int = 1152,
    modality_count: int = 3,
    num_classes: int = 2,
) -> ComputeEstimate:
    """Estimate Conv2D and dense MACs for one early-fusion forward pass."""
    if sequence_length <= 0 or modality_count <= 0 or num_classes <= 0:
        raise ValueError("Input dimensions and num_classes must be positive")

    convolution_macs = 0
    temporal_length = sequence_length
    input_channels = 1
    filters_by_block = spec.block_filters()
    for block_index, (filters, repeat_count) in enumerate(
        zip(filters_by_block, spec.conv_repeats)
    ):
        kernel_height = 2 if block_index == 0 else 1
        kernel_area = kernel_height * spec.temporal_kernel_size
        for _ in range(repeat_count):
            output_elements = modality_count * temporal_length
            if spec.convolution_type == "standard":
                convolution_macs += output_elements * filters * kernel_area * input_channels
            else:
                depthwise_macs = output_elements * input_channels * kernel_area
                pointwise_macs = output_elements * input_channels * filters
                convolution_macs += depthwise_macs + pointwise_macs
            input_channels = filters
        temporal_length //= spec.pooling_size
        if temporal_length < 1:
            raise ValueError("Pooling schedule collapses the temporal dimension")

    head_width = (
        modality_count * temporal_length * filters_by_block[-1]
        if spec.head_type == "flatten"
        else filters_by_block[-1]
    )
    return ComputeEstimate(
        convolution_macs=convolution_macs,
        dense_macs=_dense_macs(head_width, spec.dense_units, num_classes),
    )


def estimate_late_branch_compute(
    spec: LateBranchSpec,
    *,
    sequence_length: int = 1152,
    num_classes: int = 2,
) -> ComputeEstimate:
    """Estimate Conv1D and dense MACs for one late-fusion modality branch."""
    if sequence_length <= 0 or num_classes <= 0:
        raise ValueError("Input dimensions and num_classes must be positive")

    convolution_macs = 0
    temporal_length = sequence_length
    input_channels = 1
    filters_by_block = spec.block_filters()
    for filters, repeat_count in zip(filters_by_block, spec.conv_repeats):
        for _ in range(repeat_count):
            convolution_macs += (
                temporal_length
                * filters
                * spec.temporal_kernel_size
                * input_channels
            )
            input_channels = filters
        temporal_length //= 2
        if temporal_length < 1:
            raise ValueError("Pooling schedule collapses the temporal dimension")

    head_width = (
        temporal_length * filters_by_block[-1]
        if spec.head_type == "flatten"
        else filters_by_block[-1]
    )
    return ComputeEstimate(
        convolution_macs=convolution_macs,
        dense_macs=_dense_macs(head_width, spec.dense_units, num_classes),
    )


def estimate_model_compute(
    spec: ModelSpec,
    *,
    sequence_length: int = 1152,
    modality_count: int = 3,
    num_classes: int = 2,
) -> ComputeEstimate:
    """Estimate convolution and dense MACs for any PainNAS model specification."""
    if isinstance(spec, ArchitectureSpec):
        return estimate_early_fusion_compute(
            spec,
            sequence_length=sequence_length,
            modality_count=modality_count,
            num_classes=num_classes,
        )
    if modality_count != 3:
        raise ValueError("Late fusion requires exactly three modalities")
    return sum(
        (
            estimate_late_branch_compute(
                branch,
                sequence_length=sequence_length,
                num_classes=num_classes,
            )
            for branch in (spec.eda, spec.emg, spec.ecg)
        ),
        ComputeEstimate(0, 0),
    )
