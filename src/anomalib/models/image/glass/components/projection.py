# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Multi-layer projection network for feature adaptation."""

import torch
from torch import nn

from .init_weight import init_weight


class Projection(nn.Module):
    """Multi-layer linear projection with optional LeakyReLU activations.

    Args:
        in_planes: Input feature dimension.
        out_planes: Output feature dimension. Defaults to in_planes.
        n_layers: Number of linear layers.
        use_activation: Insert LeakyReLU between intermediate layers.
    """

    def __init__(
        self,
        in_planes: int,
        out_planes: int | None = None,
        n_layers: int = 1,
        use_activation: bool = False,
    ) -> None:
        super().__init__()

        if out_planes is None:
            out_planes = in_planes

        self.layers = nn.Sequential()
        for i in range(n_layers):
            in_dim = in_planes if i == 0 else out_planes
            self.layers.add_module(f"{i}fc", nn.Linear(in_dim, out_planes))
            if use_activation and i < n_layers - 1:
                self.layers.add_module(f"{i}relu", nn.LeakyReLU(0.2))

        self.apply(init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input features through the linear stack.

        Args:
            x: Input tensor of shape (B, in_planes).

        Returns:
            Projected tensor of shape (B, out_planes).
        """
        return self.layers(x)
