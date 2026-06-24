# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Decoder Layers for AnomalyVFM."""

import torch
from torch import nn


class BottleNeck(nn.Module):
    """Mask decoder bottleneck block.

    Args:
        in_dim (int): number of input channels
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        out_dim = in_dim
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.bn1 = nn.GroupNorm(32, out_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.bn2 = nn.GroupNorm(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input features.

        Args:
            x: input features

        Returns:
            (torch.Tensor): processed features
        """
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)))


class DecoderBlockBilinear(nn.Module):
    """Mask decoder block.

    Args:
        in_dim (int): number of input channels
        out_dim (int): number of output
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.bn1 = nn.GroupNorm(32, out_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.bn2 = nn.GroupNorm(32, out_dim)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input features.

        Args:
            x: input features

        Returns:
            (torch.Tensor): processed features
        """
        x = self.upsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)))


class SimpleDecoder(nn.Module):
    """Mask decoder module.

    Args:
        in_dim (int): number of input channels
        upsample_blocks (int): number of upsample blocks
        out_dim (int): number of output
    """

    def __init__(self, in_dim: int, upsample_blocks: int = 2, out_dim: int = 1) -> None:
        super().__init__()
        self.blocks = []
        ch = in_dim
        self.bot = BottleNeck(in_dim)
        for _ in range(upsample_blocks):
            self.blocks.append(DecoderBlockBilinear(ch, ch // 2))
            ch = ch // 2
        self.blocks = nn.ModuleList(self.blocks)
        self.final = nn.Conv2d(ch, out_dim, 1)
        self.final_conf = nn.Conv2d(ch, out_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict anomaly map and confidence from input tensor.

        Args:
            x: input features from the backbones

        Returns:
            (torch.Tensor): predicted anomaly map.
            (torch.Tensor): predicted confidence.
        """
        x = self.bot(x)
        for b in self.blocks:
            x = b(x)
        mask = self.final(x)
        conf = self.final_conf(x)
        return mask, conf


class SimplePredictor(nn.Module):
    """Anomaly predictor module.

    Args:
        dim (int): number of input channels
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict anomaly score from input CLS token.

        Args:
            x: input CLS token from the backbone

        Returns:
            (torch.Tensor): predicted anomaly score.
        """
        return self.lin(x)
