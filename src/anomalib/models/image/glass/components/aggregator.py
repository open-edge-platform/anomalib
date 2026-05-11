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

    def __init__(self, target_dim: int, num_features: int = 2) -> None:
        super().__init__()
        self.target_dim = target_dim
        self._onnx_kernel_size = (num_features * target_dim) // target_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns reshaped and average pooled features."""
        features = features.reshape(len(features), 1, -1)
        if torch.onnx.is_in_onnx_export():
            features = f.avg_pool1d(features, kernel_size=self._onnx_kernel_size, stride=self._onnx_kernel_size)
        else:
            features = f.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)
