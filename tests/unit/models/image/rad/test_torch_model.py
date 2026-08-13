# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RAD torch model."""

import pytest
import torch

from anomalib.models.image.rad.torch_model import RadModel


@pytest.fixture(scope="module")
def model() -> RadModel:
    """Create a RadModel with small settings for fast testing."""
    m = RadModel(
        backbone="vit_small_patch16_dinov3",
        pre_trained=False,
        layers=[3, 6, 9, 11],
        k_image=2,
        use_positional_bank=True,
        pos_radius=1,
        max_ratio=0.01,
    )
    # Build a small memory bank
    m.train()
    x = torch.randn(4, 3, 224, 224)
    m(x)
    m.build_memory_bank()
    return m


@pytest.fixture(scope="module")
def input_tensor() -> torch.Tensor:
    """Create a random input tensor."""
    return torch.randn(2, 3, 224, 224)


def test_initialization(model: RadModel) -> None:
    """Test that the model initialises without errors."""
    assert isinstance(model, RadModel)
    assert len(model.cls_banks) == 4
    assert len(model.patch_banks) == 4


def test_memory_bank_shapes(model: RadModel) -> None:
    """Test that memory bank has correct shapes."""
    for i in range(4):
        assert model.cls_banks[i].shape[0] == 4  # 4 training images
        assert model.patch_banks[i].shape[0] == 4
        assert model.cls_banks[i].ndim == 2
        assert model.patch_banks[i].ndim == 3


def test_forward_eval(model: RadModel, input_tensor: torch.Tensor) -> None:
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


def test_no_nan_in_output(model: RadModel, input_tensor: torch.Tensor) -> None:
    """Test that outputs contain no NaN values."""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    assert not torch.isnan(output.anomaly_map).any()
    assert not torch.isnan(output.pred_score).any()


def test_global_matching() -> None:
    """Test model works without positional bank."""
    m = RadModel(
        backbone="vit_small_patch16_dinov3",
        pre_trained=False,
        layers=[3, 11],
        k_image=2,
        use_positional_bank=False,
    )
    m.train()
    m(torch.randn(3, 3, 224, 224))
    m.build_memory_bank()
    m.eval()

    with torch.no_grad():
        output = m(torch.randn(1, 3, 224, 224))

    assert output.pred_score.shape == (1,)
    assert not torch.isnan(output.pred_score).any()
