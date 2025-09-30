# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Aggregates and reshapes features to a target dimension."""

import torch
import torch.nn.functional as f


class Aggregator(torch.nn.Module):
    """Aggregates and reshapes features to a target dimension.

    Input: Multi-dimensional feature tensors
    Output: Reshaped and pooled features of specified target dimension
    """

    def __init__(self, target_dim: int) -> None:
        super().__init__()
        self.target_dim = target_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns reshaped and average pooled features."""
        features = features.reshape(len(features), 1, -1)
        features = f.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)
