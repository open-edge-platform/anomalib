# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Layers needed to build DINOv2.

References:
    https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/__init__.py

Classes:
    Attention: Standard multi-head self-attention layer used in Vision Transformers.
    MemEffAttention: Memory-efficient variant of multi-head attention optimized for large inputs.
    Block: Transformer block consisting of attention, MLP, residuals, and normalization layers.
    CausalAttentionBlock: Transformer block with causal (autoregressive) attention masking.
    DINOHead: Projection head used in DINO/DINOv2 for self-supervised feature learning.
    DropPath: Implements stochastic depth, randomly dropping residual connections during training.
    LayerScale: Applies learnable per-channel scaling to stabilize deep transformer training.
    Mlp: Feedforward network used inside Vision Transformer blocks.
    PatchEmbed: Converts image patches into token embeddings for Vision Transformer inputs.
    SwiGLUFFN: SwiGLU-based feedforward network used in DINOv2 for improved expressiveness.
    SwiGLUFFNAligned: Variant of SwiGLUFFN with tensor alignment optimizations.
    SwiGLUFFNFused: Fused implementation of SwiGLUFFN for improved computational efficiency.
"""

from .attention import Attention, MemEffAttention
from .block import Block, CausalAttentionBlock
from .dino_head import DINOHead
from .drop_path import DropPath
from .layer_scale import LayerScale
from .mlp import Mlp
from .patch_embed import PatchEmbed
from .swiglu_ffn import SwiGLUFFN, SwiGLUFFNAligned, SwiGLUFFNFused

__all__ = [
    "Attention",
    "CausalAttentionBlock",
    "Block",
    "DINOHead",
    "DropPath",
    "LayerScale",
    "MemEffAttention",
    "Mlp",
    "PatchEmbed",
    "SwiGLUFFN",
    "SwiGLUFFNAligned",
    "SwiGLUFFNFused",
]
