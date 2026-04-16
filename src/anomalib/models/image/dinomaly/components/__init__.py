# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Components module for Dinomaly model.

This module provides all the necessary components for the Dinomaly Vision Transformer
architecture including layers, utilities, and vision transformer implementations.
"""

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
    # Utils
    "StableAdamW",
    "WarmCosineScheduler",
    "CosineHardMiningLoss",
    # Vision transformer
    "DinoVisionTransformer",
]
