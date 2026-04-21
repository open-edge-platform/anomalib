# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility helpers for model export."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anomalib.deploy.export import ExportType


def get_onnx_dynamo_flag(kwargs: dict[str, Any]) -> bool:
    """Return ONNX exporter dynamo flag.

    Torch 2.9 switches ``torch.onnx.export`` to ``dynamo=True`` by default.
    anomalib keeps the legacy exporter as the default because the dynamo path
    requires ``onnxscript`` and is only needed when users opt in explicitly.

    Args:
        kwargs (dict[str, Any]): Keyword arguments passed to ``torch.onnx.export``.

    Returns:
        bool: Resolved dynamo flag.
    """
    dynamo = kwargs.pop("dynamo", False)
    return False if dynamo is None else dynamo


def warn_legacy_onnx_exporter_deprecation() -> None:
    """Warn that the legacy ONNX exporter path is deprecated."""
    warnings.warn(
        "The legacy ONNX exporter path (`dynamo=False`) is deprecated and will be removed in anomalib 2.7.0. "
        "Minimum required PyTorch version will increase to 2.10 in anomalib 2.7.0. Install `anomalib[openvino]` "
        "and migrate to `dynamo=True`.",
        FutureWarning,
        stacklevel=2,
    )


def get_default_dynamic_axes(input_size: tuple[int, int] | None) -> dict[str, dict[int, str]]:
    """Build default dynamic axes for legacy ONNX export."""
    return (
        {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        if input_size
        else {"input": {0: "batch_size", 2: "height", 3: "width"}, "output": {0: "batch_size"}}
    )


def get_dynamic_shapes_from_axes(
    dynamic_axes: dict[str, dict[int, str]] | None,
    input_names: list[str],
    output_names: list[str],
) -> tuple[dict[int, str],] | None:
    """Translate single-input ``dynamic_axes`` to dynamo ``dynamic_shapes``."""
    if not dynamic_axes:
        return None

    input_name = input_names[0] if input_names else "input"
    input_axes = dynamic_axes.get(input_name)
    if input_axes is None:
        input_axes = next((axes for name, axes in dynamic_axes.items() if name not in output_names), None)
    return (dict(input_axes),) if input_axes else None


def validate_input_names(input_names: object) -> list[str]:
    """Validate ONNX input names.

    Args:
        input_names (object): Candidate input names value.

    Returns:
        list[str]: Validated input names.

    Raises:
        TypeError: If input names are not a list of strings.
    """
    if isinstance(input_names, list) and all(isinstance(name, str) for name in input_names):
        return input_names
    msg = "input_names must be a list of strings"
    raise TypeError(msg)


def validate_dynamic_axes(dynamic_axes: object) -> dict[str, dict[int, str]]:
    """Validate ONNX dynamic axes mapping.

    Args:
        dynamic_axes (object): Candidate dynamic axes value.

    Returns:
        dict[str, dict[int, str]]: Validated dynamic axes mapping.

    Raises:
        TypeError: If dynamic axes do not match expected structure.
    """
    if not isinstance(dynamic_axes, dict):
        msg = "dynamic_axes must be a dictionary of axis mappings"
        raise TypeError(msg)

    validated_axes: dict[str, dict[int, str]] = {}
    for name, axes in dynamic_axes.items():
        if not isinstance(name, str) or not isinstance(axes, dict):
            msg = "dynamic_axes must map strings to dictionaries"
            raise TypeError(msg)
        validated_axis_map: dict[int, str] = {}
        for axis, axis_name in axes.items():
            if not isinstance(axis, int) or not isinstance(axis_name, str):
                msg = "dynamic_axes axis mappings must use int keys and string values"
                raise TypeError(msg)
            validated_axis_map[axis] = axis_name
        validated_axes[name] = validated_axis_map
    return validated_axes


def raise_missing_onnxscript_error() -> None:
    """Raise actionable error for missing ``onnxscript`` dependency.

    Raises:
        ModuleNotFoundError: If ``onnxscript`` is not installed for dynamo export.
    """
    msg = (
        "ONNX export with `dynamo=True` requires the optional `onnxscript` dependency. "
        "Install `anomalib[openvino]` or `onnxscript`, or export with `dynamo=False`."
    )
    raise ModuleNotFoundError(msg)


def create_export_root(export_root: str | Path, export_type: ExportType) -> Path:
    """Create directory structure for model export.

    Args:
        export_root (str | Path): Root directory for exports.
        export_type (ExportType): Type of export (torch/onnx/openvino).

    Returns:
        Path: Created directory path.
    """
    export_root = Path(export_root) / "weights" / export_type.value
    export_root.mkdir(parents=True, exist_ok=True)
    return export_root
