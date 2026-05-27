# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the CFM model."""

from typing import cast

import pytest
import torch

from anomalib.data import Batch, InferenceBatch
from anomalib.models import CFM
from anomalib.models.image.cfm.torch_model import CFMModel


@pytest.fixture
def dummy_batch() -> Batch:
    """Creates a dummy batch with RGB and Point Cloud to test the model."""
    batch_size = 2
    image_size = 224

    # Based on the components.py implementation, both rgb and xyz
    # are expected to have an initial 2D spatial shape (B, C, H, W)
    rgb = torch.rand(batch_size, 3, image_size, image_size)
    xyz = torch.rand(batch_size, 3, image_size, image_size)

    return cast(
        "Batch",
        {
            "image": rgb,
            "point_cloud": xyz,
        },
    )


def test_cfm_model_forward_training(dummy_batch: Batch) -> None:
    """Verifies that the model correctly computes the losses during training."""
    model = CFMModel()
    model.train()  # Set the model to training mode

    tensors = cast("dict[str, torch.Tensor]", dummy_batch)
    rgb, xyz = tensors["image"], tensors["point_cloud"]

    # Execute the forward pass
    output = model(rgb, xyz)

    # Assertions
    assert isinstance(output, dict), "The output in training mode must be a dictionary"
    assert "loss" in output, "The 'loss' key is missing from the output"
    assert output["loss"].shape == torch.Size([]), "The loss must be a scalar (empty shape)"


def test_cfm_model_forward_inference(dummy_batch: Batch) -> None:
    """Verifies that the model generates anomaly maps and scores during inference."""
    model = CFMModel()
    model.eval()  # Set the model to evaluation mode

    tensors = cast("dict[str, torch.Tensor]", dummy_batch)
    rgb, xyz = tensors["image"], tensors["point_cloud"]

    with torch.no_grad():
        output = model(rgb, xyz)

    # Assertions
    assert isinstance(output, InferenceBatch), "The output in eval mode must be an Anomalib InferenceBatch"
    assert hasattr(output, "anomaly_map"), "The anomaly_map is missing from the output"
    assert hasattr(output, "pred_score"), "The pred_score is missing from the output"

    # Verify the correct shapes based on the CFMAnomalyMapGenerator
    assert output.anomaly_map.shape == (2, 1, 224, 224), "Incorrect shape for the anomaly map"
    assert output.pred_score.shape == (2,), "Incorrect shape for the pred_score"


def test_lightning_module_steps(dummy_batch: Batch) -> None:
    """Verifies that the Lightning wrapper accepts the data without crashing."""
    lightning_model = CFM()

    # Test training step
    loss = lightning_model.training_step(dummy_batch, 0)
    assert isinstance(loss, torch.Tensor), "Training step must return a tensor"
    assert loss.requires_grad, "The computation graph is broken, requires_grad is False"

    lightning_model.eval()

    with torch.no_grad():
        val_out = lightning_model.validation_step(dummy_batch, 0)
    # Test validation step
    assert "anomaly_map" in val_out, "The batch was not updated with anomaly_map"
    assert "pred_score" in val_out, "The batch was not updated with pred_score"
