# Copyright (C) 2026 Intel Corporation
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
        if torch.onnx.is_in_onnx_export():
            input_len = features.shape[-1]
            kernel_size = input_len // self.target_dim
            features = f.avg_pool1d(features, kernel_size=kernel_size, stride=kernel_size)
        else:
            features = f.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)
