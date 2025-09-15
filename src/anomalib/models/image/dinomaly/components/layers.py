# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Consolidated layer implementations for Dinomaly model.

This module contains all layer-level components used in the Dinomaly Vision Transformer
architecture, including attention mechanisms, transformer blocks, and MLP layers.

References:
    https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
    https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py
"""

import logging
from collections.abc import Callable
from typing import Any

import torch
from timm.layers.drop import DropPath
from timm.models.vision_transformer import Attention, LayerScale
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

logger = logging.getLogger("dinov2")


class MemEffAttention(Attention):
    """Memory-efficient attention from the dinov2 implementation with a small change.

    Reference:
    https://github.com/facebookresearch/dinov2/blob/592541c8d842042bb5ab29a49433f73b544522d5/dinov2/eval/segmentation_m2f/models/backbones/vit.py#L159

    Instead of using xformers's memory_efficient_attention() method, which requires adding a new dependency to anomalib,
    this implementation uses the scaled dot product from torch.
    """

    def forward(self, x: Tensor, attn_bias: Tensor | None = None) -> Tensor:
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


class LinearAttention(nn.Module):
    """Linear Attention is a Softmax-free Attention that serves as an alternative to vanilla Softmax Attention.

    As per the Dinomaly paper, using a Linear Attention leads to an "incompetence in focusing" on important
    regions related to the query, such as foreground and neighbours. This property encourages attention to spread across
    the entire image. This also contributes to computational efficiency.
    Reference :
    https://github.com/guojiajeremy/Dinomaly/blob/861a99b227fd2813b6ad8e8c703a7bea139ab735/models/vision_transformer.py#L213
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Initialize the linear attention mechanism.

        Args:
            input_dim: Input feature dimension.
            num_heads: Number of attention heads. Default: 8.
            qkv_bias: Whether to add bias to the query, key, value projections. Default: False.
            qk_scale: Override default scale factor for attention. If None, uses head_dim**-0.5. Default: None.
            attn_drop: Dropout probability for attention weights. Default: 0.0.
            proj_drop: Dropout probability for output projection. Default: 0.0.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(input_dim, input_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(input_dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass through linear attention with ELU-based feature maps.

        This implements a linear attention mechanism that avoids the quadratic complexity
        of standard attention by using ELU activation functions to create positive
        feature maps for keys and queries.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            A tuple containing:
                - Output tensor of shape (batch_size, seq_len, embed_dim).
                - Key-value interaction tensor for potential downstream use.
        """
        batch_size, seq_len, embed_dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.num_heads, embed_dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # Replace einsum operations with explicit matrix operations for OpenVINO compatibility

        kv = torch.matmul(k.transpose(-2, -1), v)

        k_sum = k.sum(dim=-2, keepdim=True)  # Shape: [..., 1, d]
        z = 1.0 / torch.sum(q * k_sum, dim=-1, keepdim=True)  # Shape: [..., s, 1]

        x = torch.matmul(q, kv) * z

        x = x.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, kv


class DinomalyMLP(nn.Module):
    """Unified MLP supporting bottleneck-style behavior, optional input dropout, and bias control.

    This can be used as a simple MLP layer or as the BottleNeck layer in Dinomaly models.
    The code is a combined class representation of several MLP implementations in the Dinomaly codebase,
    including the BottleNeck and Decoder MLPs.
    References :
    https://github.com/guojiajeremy/Dinomaly/blob/861a99b227fd2813b6ad8e8c703a7bea139ab735/models/vision_transformer.py#L67
    https://github.com/guojiajeremy/Dinomaly/blob/861a99b227fd2813b6ad8e8c703a7bea139ab735/models/vision_transformer.py#L128
    https://github.com/guojiajeremy/Dinomaly/blob/861a99b227fd2813b6ad8e8c703a7bea139ab735/dinov2/layers/mlp.py#L16

    Example usage for BottleNeck:
        >>> embedding_dim = 768
        >>> mlp = DinomalyMLP(
        ...     in_features=embedding_dim,
        ...     hidden_features=embedding_dim * 4,
        ...     out_features=embedding_dim,
        ...     drop=0.2,
        ...     bias=False,
        ...     apply_input_dropout=True)

    Example usage for a Decoder's MLP:
        >>> embedding_dim = 768
        >>> mlp = DinomalyMLP(
        ...     in_features=embedding_dim,
        ...     hidden_features=embedding_dim * 4,
        ...     drop=0.2,
        ...     bias=False,
        ...     apply_input_dropout=False)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = False,
        apply_input_dropout: bool = False,
    ) -> None:
        """Initialize the Dinomaly MLP layer.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden features. If None, defaults to in_features. Default: None.
            out_features: Number of output features. If None, defaults to in_features. Default: None.
            act_layer: Activation layer class. Default: nn.GELU.
            drop: Dropout probability. Default: 0.0.
            bias: Whether to include bias in linear layers. Default: False.
            apply_input_dropout: Whether to apply dropout to input before first linear layer. Default: False.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)
        self.apply_input_dropout = apply_input_dropout

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the MLP with optional input dropout.

        Applies the following sequence:
        1. Optional input dropout (if apply_input_dropout=True)
        2. First linear transformation
        3. Activation function
        4. Dropout
        5. Second linear transformation
        6. Final dropout

        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim).

        Returns:
            Output tensor of shape (batch_size, seq_len, out_features).
        """
        if self.apply_input_dropout:
            x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Block(nn.Module):
    """Transformer block with attention and MLP.

    The code is similar to the standard transformer block but has an extra fail-safe
    in the forward method when using memory-efficient attention.
    Reference: https://github.com/guojiajeremy/Dinomaly/blob/861a99b227fd2813b6ad8e8c703a7bea139ab735/dinov2/layers/block.py#L41
    """

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
        init_values: float | None = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = DinomalyMLP,
    ) -> None:
        """Initialize a transformer block with attention and MLP layers.

        Args:
            dim: Input feature dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of MLP hidden dimension to input dimension. Default: 4.0.
            qkv_bias: Whether to add bias to query, key, value projections. Default: False.
            proj_bias: Whether to add bias to attention output projection. Default: True.
            ffn_bias: Whether to add bias to feed-forward network layers. Default: True.
            drop: Dropout probability for MLP and projection layers. Default: 0.0.
            attn_drop: Dropout probability for attention weights. Default: 0.0.
            init_values: Initial values for layer scale. If None, layer scale is disabled. Default: None.
            drop_path: Drop path probability for stochastic depth. Default: 0.0.
            act_layer: Activation layer class for MLP. Default: nn.GELU.
            norm_layer: Normalization layer class. Default: nn.LayerNorm.
            attn_class: Attention mechanism class. Default: Attention.
            ffn_layer: Feed-forward network layer class. Default: DinomalyMLP.
        """
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
            apply_input_dropout=False,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, Any]:
        """Forward pass through the transformer block.

        Applies the standard transformer architecture:
        1. Layer normalization followed by attention
        2. Residual connection with optional layer scaling and drop path
        3. Layer normalization followed by MLP
        4. Residual connection with optional layer scaling and drop path

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            return_attention: Whether to return attention weights along with output. Default: False.

        Returns:
            If return_attention is False:
                Output tensor of shape (batch_size, seq_len, embed_dim).
            If return_attention is True:
                Tuple containing output tensor and attention weights.
                Note: Attention weights are None for MemEffAttention.
        """
        # Always use the MemEffAttention path for consistency
        if isinstance(self.attn, MemEffAttention):
            y = self.attn(self.norm1(x))
            attn = None
        else:
            y, attn = self.attn(self.norm1(x))

        x = x + self.ls1(y)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x
