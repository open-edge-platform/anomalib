# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the L2BT torch model."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from anomalib.models.image.l2bt import torch_model as l2bt_torch_model
from anomalib.models.image.l2bt.torch_model import L2BTModel


class DummyTeacher(nn.Module):
    """Mock teacher to avoid loading the heavy backbone."""

    def __init__(self, layers: list[int] | tuple[int, int]) -> None:
        """Initialize the dummy teacher."""
        super().__init__()
        del layers
        self.embed_dim = 768
        self.patch_size = 14

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mock teacher features."""
        batch_size = x.shape[0]
        n_patches = (x.shape[-1] // self.patch_size) ** 2
        middle_patch = torch.randn(batch_size, n_patches, self.embed_dim)
        last_patch = torch.randn(batch_size, n_patches, self.embed_dim)
        return middle_patch, last_patch


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> L2BTModel:
    """Create an L2BT model with a lightweight dummy teacher."""
    monkeypatch.setattr(l2bt_torch_model, "FeatureExtractor", DummyTeacher)
    return L2BTModel()


def test_training_forward(model: L2BTModel) -> None:
    """Check that training forward returns the expected loss keys."""
    model.train()
    images = torch.randn(2, 3, 224, 224)

    output = model(images)

    assert isinstance(output, dict)
    assert "loss" in output
    assert "loss_middle" in output
    assert "loss_last" in output


def test_inference_forward(model: L2BTModel) -> None:
    """Check that inference forward returns anomaly outputs."""
    model.eval()
    images = torch.randn(2, 3, 224, 224)

    output = model(images)

    assert hasattr(output, "anomaly_map")
    assert hasattr(output, "pred_score")


def test_anomaly_map_shape(model: L2BTModel) -> None:
    """Check that anomaly map and score have the expected shapes."""
    model.eval()
    images = torch.randn(1, 3, 224, 224)

    output = model(images)

    assert output.anomaly_map.shape == (1, 1, 224, 224)
    assert output.pred_score.shape == (1,)


def test_invalid_input_shape(model: L2BTModel) -> None:
    """Check that invalid input shape raises a ValueError."""
    model.eval()
    images = torch.randn(3, 224, 224)

    with pytest.raises(ValueError, match=r"Expected images with shape \(B,C,H,W\)"):
        model(images)
