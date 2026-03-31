# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PatchFlow torch model."""

import pytest
import torch

from anomalib.models.image.patchflow.torch_model import PatchflowModel


@pytest.fixture(scope="module")
def model() -> PatchflowModel:
    """Create a PatchflowModel with small settings for fast testing."""
    return PatchflowModel(
        input_size=(224, 224),
        backbone="tf_efficientnet_b5",
        pre_trained=False,
        flow_steps=1,
        flow_feature_dim=64,
        num_scales=2,
        patch_size=3,
        flow_hidden_dim=64,
    )


@pytest.fixture(scope="module")
def input_tensor() -> torch.Tensor:
    """Create a random input tensor."""
    return torch.randn(2, 3, 224, 224)


def test_initialization(model: PatchflowModel) -> None:
    """Test that the model initialises without errors."""
    assert isinstance(model, PatchflowModel)
    assert model.input_size == (224, 224)


def test_forward_train(model: PatchflowModel, input_tensor: torch.Tensor) -> None:
    """Test forward pass in training mode returns hidden_variables and jacobians."""
    model.train()
    hidden_variables, jacobians = model(input_tensor)
    assert isinstance(hidden_variables, torch.Tensor)
    assert isinstance(jacobians, torch.Tensor)
    assert hidden_variables.shape[0] == 2
    assert jacobians.shape == (2,)


def test_forward_eval(model: PatchflowModel, input_tensor: torch.Tensor) -> None:
    """Test forward pass in eval mode returns InferenceBatch."""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    assert hasattr(output, "anomaly_map")
    assert hasattr(output, "pred_score")

    assert output.anomaly_map.shape[0] == 2
    assert output.anomaly_map.shape[2] == 224
    assert output.anomaly_map.shape[3] == 224

    assert output.pred_score.shape[0] == 2


def test_no_nan_in_output(model: PatchflowModel, input_tensor: torch.Tensor) -> None:
    """Test that outputs contain no NaN values."""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    assert not torch.isnan(output.anomaly_map).any()
    assert not torch.isnan(output.pred_score).any()
