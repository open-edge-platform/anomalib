# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (C) 2025 Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0


"""DINO projection head module.

Reference:
https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/dino_head.py
"""

import torch
from torch import nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm


class DINOHead(nn.Module):
    """Projection head used in DINO and DINOv2.

    This module applies a multi-layer perceptron (MLP) followed by weight-normalized
    output projection, matching the design used in the official DINOv2 models.

    Args:
        in_dim: Input embedding dimension.
        out_dim: Output projection dimension.
        use_bn: Whether to insert BatchNorm1d layers in the MLP.
        nlayers: Number of MLP layers.
        hidden_dim: Hidden layer size for intermediate MLP layers.
        bottleneck_dim: Dimension of the final MLP output before projection.
        mlp_bias: Whether to use bias in Linear layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = False,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        mlp_bias: bool = True,
    ) -> None:
        super().__init__()

        nlayers = max(nlayers, 1)

        self.mlp: nn.Module = _build_mlp(
            nlayers=nlayers,
            in_dim=in_dim,
            bottleneck_dim=bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )

        self.apply(self._init_weights)

        self.last_layer: nn.Module = weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False),
        )
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, module: nn.Module) -> None:  # noqa: PLR6301
        """Initialize Linear layers with truncated normal weights."""
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the DINO projection head forward pass."""
        x = self.mlp(x)

        eps: float = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)

        return self.last_layer(x)


def _build_mlp(
    nlayers: int,
    in_dim: int,
    bottleneck_dim: int,
    hidden_dim: int | None = None,
    use_bn: bool = False,
    bias: bool = True,
) -> nn.Module:
    """Construct an MLP with optional batch normalization.

    Args:
        nlayers: Number of layers in the MLP.
        in_dim: Input feature dimension.
        bottleneck_dim: Output dimension of the final layer.
        hidden_dim: Hidden dimension for intermediate layers.
        use_bn: Whether to insert BatchNorm1d layers.
        bias: Whether to enable Linear layer bias.

    Returns:
        A fully constructed torch.nn.Module representing the MLP.
    """
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)

    assert hidden_dim is not None, f"hidden_dim must be provided when nlayers ({nlayers}) > 1"

    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden_dim, bias=bias),
    ]

    if use_bn:
        layers.append(nn.BatchNorm1d(hidden_dim))

    layers.append(nn.GELU())

    for _ in range(nlayers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())

    layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))

    return nn.Sequential(*layers)
