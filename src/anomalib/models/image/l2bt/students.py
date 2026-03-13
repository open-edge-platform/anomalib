# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Student projection networks used by L2BT."""

from __future__ import annotations

import torch


class FeatureProjectionMLP(torch.nn.Module):
    """MLP used as student projection network in L2BT."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_layer: type[torch.nn.Module] = torch.nn.GELU,
    ) -> None:
        """Initialize the student projection network.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            act_layer: Activation function used between layers.
        """
        super().__init__()

        hidden_dim = (in_features + out_features) // 2

        self.act_fcn = act_layer()

        self.input = torch.nn.Linear(in_features, hidden_dim)
        self.projection = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output = torch.nn.Linear(hidden_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the projection MLP.

        Args:
            x: Input feature tensor.

        Returns:
            Projected feature tensor.
        """
        x = self.input(x)
        x = self.act_fcn(x)

        x = self.projection(x)
        x = self.act_fcn(x)

        return self.output(x)
