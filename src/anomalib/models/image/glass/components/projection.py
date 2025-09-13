# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Multi-layer projection network for feature adaptation."""

import torch

from .init_weight import init_weight


class Projection(torch.nn.Module):
    """Multi-layer projection network for feature adaptation.

    Args:
        in_planes: Input feature dimension
        out_planes: Output feature dimension
        n_layers: Number of projection layers
        layer_type: Type of intermediate layers
    """

    def __init__(self, in_planes: int, out_planes: int | None = None, n_layers: int = 1, layer_type: int = 0) -> None:
        super().__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        in_ = None
        out = None
        for i in range(n_layers):
            in_ = in_planes if i == 0 else out
            out = out_planes
            self.layers.add_module(f"{i}fc", torch.nn.Linear(in_, out))
            if i < n_layers - 1 and layer_type > 1:
                self.layers.add_module(f"{i}relu", torch.nn.LeakyReLU(0.2))
        self.apply(init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the projection network to the input features.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_planes), where B is the batch size.

        Returns:
            torch.Tensor: Transformed tensor of shape (B, out_planes).
        """
        return self.layers(x)
