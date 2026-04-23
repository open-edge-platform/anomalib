# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for ONNX export mixin behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import pytest
import torch
from lightning.pytorch import LightningModule

from anomalib.models.components.base.export_mixin import ExportMixin

if TYPE_CHECKING:
    from pathlib import Path


class DummyOutput(NamedTuple):
    """Dummy named tuple output for export tests."""

    pred_score: torch.Tensor
    anomaly_map: torch.Tensor | None


class DummyExportModel(ExportMixin, LightningModule):
    """Minimal exportable Lightning module for unit tests."""

    _weight: torch.nn.Parameter

    def __init__(self) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: torch.Tensor) -> DummyOutput:
        """Return named tuple output expected by ``ExportMixin``.

        Args:
            batch (torch.Tensor): Input batch.

        Returns:
            DummyOutput: Dummy export output.
        """
        return DummyOutput(pred_score=batch.mean(dim=(1, 2, 3)) * self._weight, anomaly_map=None)


def test_to_onnx_uses_legacy_exporter_by_default(mocker: pytest.MockFixture, tmp_path: Path) -> None:
    """Test that ONNX export keeps ``dynamo=False`` by default."""
    export_mock = mocker.patch("torch.onnx.export")
    model = DummyExportModel()

    with pytest.warns(FutureWarning, match=r"(?=.*dynamo=False)(?=.*2\.10)(?=.*2\.7\.0)"):
        model.to_onnx(tmp_path, input_size=(32, 32))

    assert export_mock.call_args.kwargs["dynamo"] is False
    assert export_mock.call_args.kwargs["dynamic_axes"] == {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    assert export_mock.call_args.kwargs["dynamic_shapes"] is None


def test_to_onnx_treats_none_dynamo_as_false(mocker: pytest.MockFixture, tmp_path: Path) -> None:
    """Test that ``dynamo=None`` does not re-enable torch 2.9 default behavior."""
    export_mock = mocker.patch("torch.onnx.export")
    model = DummyExportModel()

    with pytest.warns(FutureWarning, match="dynamo=False"):
        model.to_onnx(tmp_path, input_size=(32, 32), dynamo=None)

    assert export_mock.call_args.kwargs["dynamo"] is False


def test_to_onnx_uses_dynamic_shapes_for_dynamo_export(mocker: pytest.MockFixture, tmp_path: Path) -> None:
    """Test that dynamo export receives ``dynamic_shapes`` instead of relying on conversion."""
    export_mock = mocker.patch("torch.onnx.export")
    model = DummyExportModel()

    model.to_onnx(tmp_path, input_size=(32, 32), dynamo=True)

    assert export_mock.call_args.kwargs["dynamo"] is True
    assert export_mock.call_args.kwargs["dynamic_shapes"] == ({0: "batch_size"},)
    assert export_mock.call_args.kwargs["dynamic_axes"] == {"input": {0: "batch_size"}, "output": {0: "batch_size"}}


def test_to_onnx_translates_custom_dynamic_axes_for_dynamo_export(
    mocker: pytest.MockFixture,
    tmp_path: Path,
) -> None:
    """Test that custom ``dynamic_axes`` are converted to input-only ``dynamic_shapes`` for dynamo."""
    export_mock = mocker.patch("torch.onnx.export")
    model = DummyExportModel()

    model.to_onnx(
        tmp_path,
        input_size=None,
        input_names=["image"],
        dynamo=True,
        dynamic_axes={"image": {0: "batch_size", 2: "height", 3: "width"}, "pred_score": {0: "batch_size"}},
    )

    assert export_mock.call_args.kwargs["dynamic_shapes"] == ({0: "batch_size", 2: "height", 3: "width"},)


def test_to_onnx_raises_actionable_error_for_missing_onnxscript(
    mocker: pytest.MockFixture,
    tmp_path: Path,
) -> None:
    """Test that dynamo export failures mention ``onnxscript`` remediation."""
    model = DummyExportModel()
    export_mock = mocker.patch("torch.onnx.export")
    export_mock.side_effect = ModuleNotFoundError("No module named 'onnxscript'")

    with pytest.raises(ModuleNotFoundError, match="onnxscript") as exception:
        model.to_onnx(tmp_path, input_size=(32, 32), dynamo=True)

    assert "dynamo=False" in str(exception.value)
