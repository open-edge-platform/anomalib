# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PatchFlow Anomaly Map Generator Implementation.

This module generates anomaly heatmaps from the single fused hidden variable
tensor produced by the PatchFlow normalizing flow.
"""

import torch
from omegaconf import ListConfig
from torch import nn
from torch.nn import functional as F  # noqa: N812


class AnomalyMapGenerator(nn.Module):
    """Generate anomaly heatmap from PatchFlow hidden variables.

    Unlike FastFlow which averages maps from multiple flow blocks, PatchFlow
    operates on a single fused tensor.

    Args:
        input_size (ListConfig | tuple): Target size ``(height, width)`` for the
            anomaly map.
    """

    def __init__(self, input_size: ListConfig | tuple) -> None:
        super().__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else tuple(input_size)

    def forward(self, hidden_variables: torch.Tensor) -> torch.Tensor:
        """Generate anomaly heatmap from hidden variables.

        Args:
            hidden_variables (torch.Tensor): Hidden variable tensor of shape
                ``(N, C, H, W)`` from the normalizing flow.

        Returns:
            torch.Tensor: Anomaly heatmap of shape ``(N, 1, H, W)``.
        """
        log_prob = -torch.mean(hidden_variables**2, dim=1, keepdim=True) * 0.5
        prob = torch.exp(log_prob)
        return F.interpolate(
            input=-prob,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
        )
