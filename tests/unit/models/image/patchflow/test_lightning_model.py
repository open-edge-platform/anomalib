# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PatchFlow lightning model."""

from unittest.mock import MagicMock

import pytest
import torch

from anomalib.data import ImageBatch
from anomalib.models.image.patchflow import Patchflow


@pytest.fixture(scope="module")
def model() -> Patchflow:
    """Create a Patchflow lightning module with small settings."""
    return Patchflow(
        backbone="resnet18",
        pre_trained=False,
        flow_steps=1,
        flow_feature_dim=64,
        num_scales=2,
        patch_size=3,
        flow_hidden_dim=64,
    )


@pytest.fixture(scope="module")
def batch() -> ImageBatch:
    """Create a fake batch."""
    return ImageBatch(image=torch.randn(2, 3, 256, 256))


def test_initialization(model: Patchflow) -> None:
    """Test that the lightning module initialises."""
    assert isinstance(model, Patchflow)
    assert model.learning_type.name == "ONE_CLASS"


def test_training_step(model: Patchflow, batch: ImageBatch) -> None:
    """Test that training_step returns a dict with a scalar loss."""
    model.log = MagicMock()
    model.model.train()
    output = model.training_step(batch)

    assert isinstance(output, dict)
    assert "loss" in output
    assert output["loss"].dim() == 0  # scalar
    assert not torch.isnan(output["loss"])


def test_validation_step(model: Patchflow, batch: ImageBatch) -> None:
    """Test that validation_step returns predictions."""
    model.model.eval()
    with torch.no_grad():
        output = model.validation_step(batch)

    assert hasattr(output, "pred_score")
    assert hasattr(output, "anomaly_map")
    assert output.pred_score is not None
    assert output.anomaly_map is not None
