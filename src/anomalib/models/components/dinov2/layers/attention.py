# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (C) 2025 Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Attention layers for DINOv2 Vision Transformers.

This module provides:
- A standard multi-head self-attention implementation (`Attention`)
- A memory-efficient xFormers-based version (`MemEffAttention`) when xFormers is available

These layers are used as core components within DINOv2 and Dinomaly transformer
blocks for feature extraction and masked modeling.
"""

import logging

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    """Standard multi-head self-attention layer.

    Implements a QKV-projection attention block with optional bias, dropout, and
    projection layers. This is the default attention mechanism used in DINOv2
    when memory-efficient attention kernels are not available.

    Args:
        dim: Embedding dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether to include bias in the QKV projections.
        proj_bias: Whether to include bias in the output projection.
        attn_drop: Dropout probability applied to attention weights.
        proj_drop: Dropout probability applied after projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self,
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        factor: float = 1.0,
    ) -> None:
        """Initialize QKV and projection weights.

        Args:
            init_attn_std: Standard deviation for attention weights.
            init_proj_std: Standard deviation for projection weights.
            factor: Additional scaling factor for projection initialization.
        """
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or (init_attn_std * factor)

        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)

        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        """Apply multi-head self-attention.

        Args:
            x: Input sequence of shape ``(B, N, C)``.
            is_causal: If True, applies causal masking.

        Returns:
            torch.Tensor of shape ``(B, N, C)`` containing attended features.
        """
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        x = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=is_causal,
        )

        x = x.transpose(1, 2).contiguous().view(b, n, c)
        return self.proj_drop(self.proj(x))


class MemEffAttention(Attention):
    """Memory-efficient attention from the dinov2 implementation with a small change.

    Reference:
    https://github.com/facebookresearch/dinov2/blob/592541c8d842042bb5ab29a49433f73b544522d5/dinov2/eval/segmentation_m2f/models/backbones/vit.py#L159

    Instead of using xformers's memory_efficient_attention() method, which requires adding a new dependency to anomalib,
    this implementation uses the scaled dot product from torch.
    """

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Compute memory-efficient attention using PyTorch's scaled dot product attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
            attn_bias: Optional attention bias mask. Default: None.

        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, embed_dim // self.num_heads)

        q, k, v = qkv.unbind(2)

        # Use PyTorch's native scaled dot product attention for memory efficiency.
        # Replaced xformers's memory_efficient_attention() method with pytorch's scaled
        # dot product.
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_bias,
        )
        x = x.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)

        x = self.proj(x)
        return self.proj_drop(x)
