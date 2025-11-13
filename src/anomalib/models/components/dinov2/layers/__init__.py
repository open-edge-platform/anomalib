# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Layers needed to build DINOv2.

References:
https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/__init__.py
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
