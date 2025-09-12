# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Components module for Dinomaly model.

This module provides all the necessary components for the Dinomaly Vision Transformer
architecture including layers, model loader, utilities, and vision transformer implementations.
"""

# Model loader
from .dinov2_loader import DinoV2Loader, load

# Layer components
from .layers import Block, DinomalyMLP, LinearAttention, MemEffAttention

# Training-related classes: Loss, Optimizer and scheduler
from .loss import CosineHardMiningLoss
from .optimizer import StableAdamW, WarmCosineScheduler

# Vision transformer components
from .vision_transformer import DinoVisionTransformer

__all__ = [
    # Layers
    "Block",
    "DinomalyMLP",
    "LinearAttention",
    "MemEffAttention",
    # Model loader
    "DinoV2Loader",
    "load",
    # Utils
    "StableAdamW",
    "WarmCosineScheduler",
    "CosineHardMiningLoss",
    # Vision transformer
    "DinoVisionTransformer",
]
