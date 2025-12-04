# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (C) 2025 Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Stochastic depth drop-path implementation used in DINOv2.

This module provides a functional drop-path operation and a corresponding
nn.Module wrapper. Drop-path (also known as stochastic depth) randomly
drops entire residual branches during training to improve model robustness.
"""

import torch
from torch import nn


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Apply stochastic depth to an input tensor.

    Args:
        x: Input tensor to process.
        drop_prob: Probability of dropping the path.
        training: Whether the module is in training mode.

    Returns:
        torch.Tensor with dropped paths applied during training, or the original
        tensor during evaluation.

    Notes:
        Drop-path randomly zeroes the entire residual branch for each sample
        in the batch while scaling the remaining samples appropriately.
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)

    return x * random_tensor


class DropPath(nn.Module):
    """Stochastic depth module for residual blocks.

    Applies drop-path per sample. During training, residual branches are
    randomly removed with probability ``drop_prob`` while scaling the
    remaining paths. In evaluation mode, the module becomes a no-op.

    Args:
        drop_prob: Probability of dropping a path.
    """

    def __init__(self, drop_prob: float | None = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob if drop_prob is not None else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying stochastic depth."""
        return drop_path(x, drop_prob=self.drop_prob, training=self.training)
