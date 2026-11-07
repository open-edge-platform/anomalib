# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SuperSimpleNet."""

import pytest
import torch

from anomalib.models.image.supersimplenet.torch_model import SupersimplenetModel


@pytest.fixture(scope="module")
def get_inputs() -> dict[str, torch.Tensor]:
    """Fixture returning mock input tensors."""
    return {
        "image": torch.rand(2, 3, 128, 128),
        "label": torch.zeros(2),
        "mask": torch.rand(2, 128, 128),
    }


@pytest.mark.parametrize(
    ("adapt_cls_feat"),
    [True, False],
)
def test_cls_feat_adapt(adapt_cls_feat: bool, get_inputs: dict[str, torch.Tensor]) -> None:
    """Test that both ICPR and JIMS version of cls feature paths work."""
    model = SupersimplenetModel(adapt_cls_features=adapt_cls_feat)
    mock_dict = get_inputs
    model.forward(images=mock_dict["image"], masks=mock_dict["mask"], labels=mock_dict["label"])


def test_no_mask(get_inputs: dict[str, torch.Tensor]) -> None:
    """Test that model works without masks provided."""
    model = SupersimplenetModel()
    mock_dict = get_inputs
    model.forward(images=mock_dict["image"], labels=mock_dict["label"])


def test_fail_anomalous_no_mask(get_inputs: dict[str, torch.Tensor]) -> None:
    """Test that model fails when anomalous samples are provided without masks."""
    model = SupersimplenetModel()
    mock_dict = get_inputs
    with pytest.raises(
        RuntimeError,
        match=r"Training with anomalous samples without GT masks is currently not supported",
    ):
        model.forward(images=mock_dict["image"], labels=torch.tensor([0, 1]))
