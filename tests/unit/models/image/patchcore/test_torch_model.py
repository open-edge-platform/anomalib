# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PatchCore torch model."""

import math
from pathlib import Path

import pytest
import torch

from anomalib.data.utils.tiler import Tiler
from anomalib.models.image.patchcore import torch_model
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


@pytest.fixture(params=[False, True], ids=["untiled", "tiled"])
def model(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> PatchcoreModel:
    """PatchCore with a random memory bank and a chunk size smaller than one image's patches."""
    # A 64x64 image gives 8x8 = 64 patches, so eager inference searches the memory bank in several chunks.
    monkeypatch.setattr(torch_model, "DEFAULT_CHUNK_SIZE", 16)
    torch.manual_seed(0)
    model = PatchcoreModel(backbone="resnet18", layers=["layer2", "layer3"], pre_trained=False, num_neighbors=3)
    model.memory_bank = torch.randn(32, 384)  # resnet18 layer2 + layer3 channels
    if request.param:
        model.tiler = Tiler(tile_size=32, stride=32)
    return model.eval()


def test_export_keeps_batch_size_dynamic(model: PatchcoreModel) -> None:
    """Export searches all patches at once, so any batch size runs and matches chunked eager inference."""
    batch_size = torch.export.Dim("batch_size")
    exported = torch.export.export(model, (torch.rand(2, 3, 64, 64),), dynamic_shapes=({0: batch_size},)).module()

    for size in (1, 3):
        images = torch.rand(size, 3, 64, 64)
        actual, expected = exported(images), model(images)
        torch.testing.assert_close(actual.pred_score, expected.pred_score)
        torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map)


def test_onnx_export_single_memory_bank(tmp_path: Path) -> None:
    """Exported ONNX model keeps the memory bank once without duplicate transposed copy."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    torch.manual_seed(0)
    model = PatchcoreModel(backbone="resnet18", layers=["layer2", "layer3"], pre_trained=False, num_neighbors=3)
    model.memory_bank = torch.randn(32, 384)
    model.eval()

    images = torch.rand(1, 3, 64, 64)
    onnx_path = tmp_path / "model.onnx"
    opt_path = tmp_path / "optimized.onnx"

    torch.onnx.export(model, (images,), str(onnx_path), dynamo=True)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    session_options.optimized_model_filepath = str(opt_path)
    session = ort.InferenceSession(str(onnx_path), session_options, providers=["CPUExecutionProvider"])

    optimized_model = onnx.load(str(opt_path), load_external_data=False)
    bank_size = model.memory_bank.numel()
    bank_sized_inits = [init.name for init in optimized_model.graph.initializer if math.prod(init.dims) == bank_size]
    assert len(bank_sized_inits) == 1
    assert bank_sized_inits[0] == "memory_bank"

    with torch.no_grad():
        expected = model(images)
    actual_score, actual_map = session.run(None, {"input_tensor": images.numpy()})

    torch.testing.assert_close(torch.from_numpy(actual_score), expected.pred_score, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.from_numpy(actual_map), expected.anomaly_map, atol=1e-5, rtol=1e-5)


def test_memory_bank_norm_backward_compatibility() -> None:
    """Loading a state dict without memory_bank_norm maintains buffer parity."""
    model = PatchcoreModel(backbone="resnet18", layers=["layer2", "layer3"], pre_trained=False)
    bank = torch.randn(16, 384)
    model.load_state_dict({"memory_bank": bank}, strict=False)
    torch.testing.assert_close(model.memory_bank_norm, bank.pow(2).sum(dim=-1))
