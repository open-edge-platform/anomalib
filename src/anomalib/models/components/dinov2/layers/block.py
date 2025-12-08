# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (C) 2025 Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0


"""Transformer blocks used in DINOv2 Vision Transformers.

This module implements:
- Standard transformer blocks with attention and MLP (`Block`)
- Causal attention blocks (`CausalAttentionBlock`)
"""

from collections.abc import Callable

import torch
from torch import nn

from .attention import Attention
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp


class Block(nn.Module):
    """Standard transformer block with attention and MLP.

    This block applies layer normalization, multi-head self-attention, optional
    layer scaling, stochastic depth, and a feed-forward network.

    Args:
        dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Expansion ratio for the MLP hidden dimension.
        qkv_bias: Whether to use bias in the QKV projections.
        proj_bias: Whether to use bias in the attention projection.
        ffn_bias: Whether to use bias in the MLP.
        drop: Dropout probability applied to projections and MLP.
        attn_drop: Dropout probability applied to attention weights.
        init_values: Initial value for LayerScale. If ``None``, LayerScale is disabled.
        drop_path: Stochastic depth rate.
        act_layer: Activation layer factory for the MLP.
        norm_layer: Normalization layer factory.
        attn_class: Attention layer factory.
        ffn_layer: Feed-forward layer factory.
    """

    # Threshold for using optimized stochastic depth implementation
    STOCHASTIC_DEPTH_THRESHOLD = 0.1

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | torch.Tensor | None = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP residual blocks with optional stochastic depth."""

        def attn_residual_func(inp: torch.Tensor) -> torch.Tensor:
            return self.ls1(self.attn(self.norm1(inp)))

        def ffn_residual_func(inp: torch.Tensor) -> torch.Tensor:
            return self.ls2(self.mlp(self.norm2(inp)))

        if self.training and self.sample_drop_ratio > self.STOCHASTIC_DEPTH_THRESHOLD:
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x))
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x)
            x = x + ffn_residual_func(x)
        return x


class CausalAttentionBlock(nn.Module):
    """Transformer block with causal attention.

    This block applies causal self-attention followed by a feed-forward network,
    with optional LayerScale and dropout.

    Args:
        dim: Embedding dimension.
        num_heads: Number of attention heads.
        ffn_ratio: Expansion ratio for the feed-forward network.
        ls_init_value: Initial value for LayerScale. If ``None``, LayerScale is disabled.
        is_causal: Whether to apply causal masking.
        act_layer: Activation layer factory for the MLP.
        norm_layer: Normalization layer factory.
        dropout_prob: Dropout probability applied to attention and MLP.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: float | None = None,
        is_causal: bool = True,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.is_causal = is_causal
        self.ls1 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()
        self.attention_norm = norm_layer(dim)
        self.attention = Attention(
            dim,
            num_heads,
            attn_drop=dropout_prob,
            proj_drop=dropout_prob,
        )

        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()

    def init_weights(
        self,
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        init_fc_std: float | None = None,
        factor: float = 1.0,
    ) -> None:
        """Initialize attention and MLP weights."""
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or (init_attn_std * factor)
        init_fc_std = init_fc_std or (2 * self.dim) ** -0.5

        self.attention.init_weights(init_attn_std, init_proj_std)
        self.attention_norm.reset_parameters()
        nn.init.normal_(self.feed_forward.fc1.weight, std=init_fc_std)
        nn.init.normal_(self.feed_forward.fc2.weight, std=init_proj_std)
        self.ffn_norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal attention followed by a feed-forward block."""
        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        return x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))


def drop_add_residual_stochastic_depth(
    x: torch.Tensor,
    residual_func: Callable[[torch.Tensor], torch.Tensor],
    sample_drop_ratio: float = 0.0,
) -> torch.Tensor:
    """Apply stochastic depth to a residual branch on a subset of samples.

    Args:
        x: Input tensor of shape ``(B, N, C)``.
        residual_func: Function computing the residual on a subset of samples.
        sample_drop_ratio: Fraction of samples to drop for residual computation.

    Returns:
        torch.Tensor with residual added to a subset of samples.
    """
    b, _, _ = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = torch.randperm(b, device=x.device)[:sample_subset_size]
    x_subset = x[brange]

    residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual_flat = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    x_plus_residual = torch.index_add(
        x_flat,
        0,
        brange,
        residual_flat.to(dtype=x.dtype),
        alpha=residual_scale_factor,
    )
    return x_plus_residual.view_as(x)
