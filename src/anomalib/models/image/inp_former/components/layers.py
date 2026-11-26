# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Consolidated layer implementations for INP-Former model.

This module contains all layer-level components used in the INP-Former Vision Transformer
architecture, including aggreation blocks and prototype blocks.

References:
    https://github.com/luow23/INP-Former/blob/main/models/vision_transformer.py
"""

import logging

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components.dinov2.layers import DropPath
from anomalib.models.image.dinomaly.components import DinomalyMLP

logger = logging.getLogger("dinov2")


class AggregationAttention(nn.Module):
    """Cross-attention used by the INP Extractor to aggregate patch tokens into INPs.

    Standard scaled dot-product cross-attention where the query comes from a set of
    learnable prototype tokens and the keys/values come from the encoder patch tokens.
    The output is a linear combination of patch tokens for each query, producing the
    Intrinsic Normal Prototypes (INPs) that summarize the image's content.

    Args:
        dim (int): Channel dimension of the input tokens.
        num_heads (int): Number of attention heads. Defaults to ``8``.
        qkv_bias (bool): If ``True``, adds a learnable bias to the q/kv projections.
            Defaults to ``False``.
        qk_scale (float | None): Override for the default ``head_dim ** -0.5`` scaling
            applied to the attention logits. Defaults to ``None``.
        attn_drop (float): Dropout probability applied to the attention weights.
            Defaults to ``0.0``.
        proj_drop (float): Dropout probability applied to the output projection.
            Defaults to ``0.0``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass for the aggregation attention."""
        batch_size, seq_len, embed_dim = x.shape
        _, n, _ = y.shape
        q = (
            self.q(x)
            .reshape(batch_size, seq_len, 1, self.num_heads, embed_dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)[0]
        )
        kv = self.kv(y).reshape(batch_size, n, 2, self.num_heads, embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attnmap = attn.softmax(dim=-1)
        attn = self.attn_drop(attnmap)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        return self.proj_drop(x)


class AggregationBlock(nn.Module):
    """Transformer block wrapping :class:`AggregationAttention` with an MLP.

    Used to build the INP Extractor: the queries ``x`` (learnable prototype tokens) are
    updated by cross-attending to the keys/values ``y`` (encoder patch tokens),
    followed by a feed-forward MLP. Both sub-layers use pre-norm residual connections
    and optional stochastic depth.

    Args:
        dim (int): Channel dimension of the input tokens.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Hidden-to-input ratio for the MLP. Defaults to ``4.0``.
        qkv_bias (bool): If ``True``, enables bias in the q/kv projections.
            Defaults to ``False``.
        qk_scale (float | None): Override for the default attention scaling.
            Defaults to ``None``.
        drop (float): Dropout probability for the MLP and projections.
            Defaults to ``0.0``.
        attn_drop (float): Dropout probability for the attention weights.
            Defaults to ``0.0``.
        drop_path (float): Stochastic depth rate. Defaults to ``0.0``.
        act_layer (nn.Module): Activation used in the MLP. Defaults to ``nn.GELU``.
        norm_layer (nn.Module): Normalization layer. Defaults to ``nn.LayerNorm``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AggregationAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DinomalyMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass for the aggregation block."""
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(y)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PrototypeAttention(nn.Module):
    """INP-Guided cross-attention used inside the decoder.

    Implements the INP-Guided Attention from the INP-Former paper: the query comes
    from the decoder patch tokens and the keys/values come from the Intrinsic Normal
    Prototypes (INPs). Query and key are L2-normalized so the attention logits behave
    like cosine similarities, scaled by a per-head learnable temperature, and passed
    through a ReLU instead of a softmax to suppress weak correlations and noise. The
    output is constrained to lie in the span of the INPs, which suppresses the
    reconstruction of anomalous queries.

    Args:
        dim (int): Channel dimension of the input tokens.
        num_heads (int): Number of attention heads. Defaults to ``8``.
        qkv_bias (bool): If ``True``, adds a learnable bias to the q/kv projections.
            Defaults to ``False``.
        attn_drop (float): Dropout probability applied to the attention weights.
            Defaults to ``0.0``.
        proj_drop (float): Dropout probability applied to the output projection.
            Defaults to ``0.0``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The attended output tokens and the
        attention map (before dropout it is post-ReLU; the returned tensor is the
        one used to weight the values).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.learn_scale = nn.Parameter(torch.ones(num_heads, 1, 1), requires_grad=True)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, prototype_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the prototype attention."""
        batch_size, seq_len, embed_dim = x.shape
        prototype_num = prototype_token.shape[1]
        q = (
            self.q(x)
            .reshape(batch_size, seq_len, 1, self.num_heads, embed_dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)[0]
        )
        kv = (
            self.kv(prototype_token)
            .reshape(batch_size, prototype_num, 2, self.num_heads, embed_dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.learn_scale
        attn = F.relu(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class PrototypeBlock(nn.Module):
    """Transformer block forming one layer of the INP-Guided Decoder.

    Wraps :class:`PrototypeAttention` with a feed-forward MLP. Following the
    INP-Former paper, the first residual connection around the attention sub-layer
    is removed so that anomalous query features cannot bypass the INP-guided
    attention and leak directly into the reconstruction; the MLP sub-layer keeps its
    standard pre-norm residual connection.

    Args:
        dim (int): Channel dimension of the input tokens.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Hidden-to-input ratio for the MLP. Defaults to ``4.0``.
        qkv_bias (bool): If ``True``, enables bias in the q/kv projections.
            Defaults to ``False``.
        drop (float): Dropout probability for the MLP and projections.
            Defaults to ``0.0``.
        attn_drop (float): Dropout probability for the attention weights.
            Defaults to ``0.0``.
        drop_path (float): Stochastic depth rate. Defaults to ``0.0``.
        act_layer (nn.Module): Activation used in the MLP. Defaults to ``nn.GELU``.
        norm_layer (nn.Module): Normalization layer. Defaults to ``nn.LayerNorm``.

    Forward Args:
        x (torch.Tensor): Decoder patch tokens of shape ``(batch_size, seq_len, embed_dim)`` used as queries.
        prototype (torch.Tensor): Intrinsic Normal Prototypes of shape ``(batch_size, prototype_num, embed_dim)``
            used as keys/values.
        return_attention (bool): If ``True``, also return the attention map from the
            INP-Guided Attention. Defaults to ``False``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PrototypeAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DinomalyMLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        prototype: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the prototype block."""
        y, attn = self.attn(self.norm1(x), self.norm1(prototype))
        x = self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x
