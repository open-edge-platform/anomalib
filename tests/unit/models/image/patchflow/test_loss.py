# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PatchFlow loss."""

import torch

from anomalib.models.image.patchflow.loss import PatchflowLoss


def test_forward() -> None:
    """Test loss computation with fake inputs."""
    loss_fn = PatchflowLoss()
    hidden_variables = torch.randn(2, 64, 8, 8)
    jacobians = torch.randn(2)

    loss = loss_fn(hidden_variables, jacobians)

    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)


def test_output_is_finite() -> None:
    """Test loss output is finite."""
    loss_fn = PatchflowLoss()
    hidden_variables = torch.randn(4, 128, 16, 16)
    jacobians = torch.randn(4)

    loss = loss_fn(hidden_variables, jacobians)

    assert torch.isfinite(loss)
