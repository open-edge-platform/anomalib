# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the CFM model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models import CFM
from anomalib.models.image.cfm.torch_model import CFMModel

if TYPE_CHECKING:
    from collections.abc import Callable


class DummyBatch:
    """Simple multimodal batch used for offline CFM tests."""

    def __init__(self, image: torch.Tensor, point_cloud: torch.Tensor) -> None:
        self.image: torch.Tensor = image
        self.point_cloud: torch.Tensor = point_cloud
        self.anomaly_map: torch.Tensor | None = None
        self.pred_score: torch.Tensor | None = None

    def update(self, **kwargs: torch.Tensor) -> DummyBatch:
        """Update batch fields in-place and return self."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self


@pytest.fixture(autouse=True)
def mock_cfm_backbones(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch DINO and Point-MAE to avoid network access."""

    class MockDinoBackbone:
        def __init__(self) -> None:
            self.blocks: nn.Sequential = nn.Sequential(*[nn.Identity() for _ in range(12)])

        @staticmethod
        def forward_features(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 785, 768, device=x.device, dtype=x.dtype)

    def create_model(*_args: object, **_kwargs: object) -> MockDinoBackbone:
        del _args, _kwargs
        return MockDinoBackbone()

    def resolve_pointmae_weights(*_args: object, **_kwargs: object) -> Path:
        del _args, _kwargs
        return Path("nonexistent_pointmae_weights.pth")

    monkeypatch.setattr("anomalib.models.image.cfm.components.timm.create_model", create_model)
    monkeypatch.setattr("anomalib.models.image.cfm.components._resolve_pointmae_weights", resolve_pointmae_weights)


@pytest.fixture
def dummy_batch() -> DummyBatch:
    """Creates a dummy batch with RGB and Point Cloud to test the model."""
    batch_size = 2
    image_size = 224

    rgb = torch.rand(batch_size, 3, image_size, image_size)
    xyz = torch.rand(batch_size, 3, image_size, image_size)

    return DummyBatch(image=rgb, point_cloud=xyz)


def test_cfm_model_forward_training(dummy_batch: DummyBatch) -> None:
    """Verifies that the model correctly computes the losses during training."""
    model = CFMModel()
    _ = model.train()

    output: dict[str, torch.Tensor] = cast("dict[str, torch.Tensor]", model(dummy_batch.image, dummy_batch.point_cloud))

    assert isinstance(output, dict), "The output in training mode must be a dictionary"
    assert "loss" in output, "The 'loss' key is missing from the output"
    assert output["loss"].shape == torch.Size([]), "The loss must be a scalar (empty shape)"


def test_cfm_model_forward_inference(dummy_batch: DummyBatch) -> None:
    """Verifies that the model generates anomaly maps and scores during inference."""
    model = CFMModel()
    _ = model.eval()

    with torch.no_grad():
        output: InferenceBatch = cast("InferenceBatch", model(dummy_batch.image, dummy_batch.point_cloud))

    assert isinstance(output, InferenceBatch), "The output in eval mode must be an Anomalib InferenceBatch"
    assert hasattr(output, "anomaly_map"), "The anomaly_map is missing from the output"
    assert hasattr(output, "pred_score"), "The pred_score is missing from the output"

    anomaly_map = output.anomaly_map
    pred_score = output.pred_score
    assert anomaly_map is not None
    assert pred_score is not None
    assert anomaly_map.shape == (2, 1, 224, 224), "Incorrect shape for the anomaly map"
    assert pred_score.shape == (2,), "Incorrect shape for the pred_score"


def test_lightning_module_steps(dummy_batch: DummyBatch) -> None:
    """Verifies that the Lightning wrapper accepts the data without crashing."""
    lightning_model = CFM()

    training_step = cast("Callable[[DummyBatch, int], torch.Tensor]", lightning_model.training_step)
    loss = training_step(dummy_batch, 0)
    assert isinstance(loss, torch.Tensor), "Training step must return a tensor"
    assert loss.requires_grad, "The computation graph is broken, requires_grad is False"

    _ = lightning_model.eval()

    with torch.no_grad():
        validation_step = cast("Callable[[DummyBatch, int], DummyBatch]", lightning_model.validation_step)
        val_out = validation_step(dummy_batch, 0)

    assert isinstance(val_out, DummyBatch)
    assert "anomaly_map" in val_out.__dict__, "The batch was not updated with anomaly_map"
    assert "pred_score" in val_out.__dict__, "The batch was not updated with pred_score"
