# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (C) 2025 Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

"""SwiGLU-based feed-forward layers used in DINOv2.

This module provides multiple variants of SwiGLU feed-forward networks,
including:
- A pure PyTorch implementation (`SwiGLUFFN`)
- A fused xFormers version when available (`SwiGLUFFNFused`)
- An aligned variant for memory efficiency (`SwiGLUFFNAligned`)

These layers are used as transformer FFN blocks in DINOv2 models.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class SwiGLUFFN(nn.Module):
    """Pure PyTorch SwiGLU feed-forward network.

    This network computes:
        hidden = silu(W1(x)) * W2(x)
        output = W3(hidden)

    Args:
        in_features: Input feature dimension.
        hidden_features: Hidden layer dimension (defaults to ``in_features``).
        out_features: Output feature dimension (defaults to ``in_features``).
        act_layer: Unused placeholder to mimic MLP API.
        drop: Unused dropout placeholder for API compatibility.
        bias: Whether to use bias in linear layers.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] | None = None,  # noqa: ARG002
        drop: float = 0.0,  # noqa: ARG002
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU feed-forward transformation."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class SwiGLUFFNFused(SwiGLUFFN):
    """Fused SwiGLU FFN using xFormers when available.

    This implementation reduces memory usage by aligning hidden dimensions
    and delegating the SwiGLU computation to optimized xFormers kernels.

    Args:
        in_features: Input feature dimension.
        hidden_features: Hidden layer dimension (defaults to ``in_features``).
        out_features: Output feature dimension (defaults to ``in_features``).
        act_layer: Unused placeholder for API compatibility.
        drop: Unused dropout placeholder.
        bias: Whether linear layers use bias.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] | None = None,  # noqa: ARG002
        drop: float = 0.0,  # noqa: ARG002
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Align hidden dimension for fused kernels
        hidden_aligned = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        super().__init__(
            in_features=in_features,
            hidden_features=hidden_aligned,
            out_features=out_features,
            bias=bias,
        )


class SwiGLUFFNAligned(nn.Module):
    """SwiGLU FFN with explicit alignment for hardware efficiency.

    Args:
        in_features: Input feature dimension.
        hidden_features: Hidden layer dimension (defaults to ``in_features``).
        out_features: Output feature dimension (defaults to ``in_features``).
        act_layer: Activation layer (unused; API compatibility).
        drop: Dropout (unused).
        bias: Whether linear layers use bias.
        align_to: Alignment multiple for hidden dimension.
        device: Optional device for parameter initialization.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,  # noqa: ARG002
        drop: float = 0.0,  # noqa: ARG002
        bias: bool = True,
        align_to: int = 8,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        d = int(hidden_features * 2 / 3)
        hidden_aligned = d + (-d % align_to)

        self.w1 = nn.Linear(in_features, hidden_aligned, bias=bias, device=device)
        self.w2 = nn.Linear(in_features, hidden_aligned, bias=bias, device=device)
        self.w3 = nn.Linear(hidden_aligned, out_features, bias=bias, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply aligned SwiGLU feed-forward transformation."""
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)
