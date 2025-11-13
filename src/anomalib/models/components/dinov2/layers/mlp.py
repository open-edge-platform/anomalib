# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Feed-forward MLP block used in DINOv2 Vision Transformers.

This module implements the standard 2-layer transformer MLP block with an
activation function and dropout. It is used as the feed-forward component
inside each transformer block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Callable


class Mlp(nn.Module):
    """Two-layer feed-forward MLP used inside transformer blocks.

    Args:
        in_features: Input feature dimension.
        hidden_features: Dimension of the hidden expansion layer. Defaults to
            ``in_features`` when ``None``.
        out_features: Output feature dimension. Defaults to ``in_features`` when
            ``None``.
        act_layer: Activation layer constructor.
        drop: Dropout probability applied after each layer.
        bias: Whether linear layers use bias terms.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the two-layer feed-forward transformation."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)
