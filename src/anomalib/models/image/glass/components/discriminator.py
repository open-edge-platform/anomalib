# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Discriminator network for anomaly detection."""

import torch

from .init_weight import init_weight


class Discriminator(torch.nn.Module):
    """Discriminator network for anomaly detection.

    Args:
        in_planes: Input feature dimension
        n_layers: Number of layers
        hidden: Hidden layer dimensions
    """

    def __init__(self, in_planes: int, n_layers: int = 2, hidden: int | None = None) -> None:
        super().__init__()

        hidden_ = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            in_ = in_planes if i == 0 else hidden_
            hidden_ = int(hidden_ // 1.5) if hidden is None else hidden
            self.body.add_module(
                f"block{i + 1}",
                torch.nn.Sequential(
                    torch.nn.Linear(in_, hidden_),
                    torch.nn.BatchNorm1d(hidden_),
                    torch.nn.LeakyReLU(0.2),
                ),
            )
        self.tail = torch.nn.Sequential(
            torch.nn.Linear(hidden_, 1, bias=False),
            torch.nn.Sigmoid(),
        )
        self.apply(init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the discriminator network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_planes), where B is the batch size.

        Returns:
            torch.Tensor: Output tensor of shape (B, 1) containing probability scores.
        """
        x = self.body(x)
        return self.tail(x)
