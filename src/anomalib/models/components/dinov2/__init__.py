# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Anomalib's Vision Transformer implementation.

References:
https://github.com/facebookresearch/dinov2/blob/main/dinov2/

Classes:
    DinoVisionTransformer: DINOv2 implementation.
    DinoV2Loader: Loader class to support downloading and loading weights.
"""

# vision transformer
# loader
from .dinov2_loader import DinoV2Loader
from .vision_transformer import (
    DinoVisionTransformer,
    vit_base,
    vit_giant2,
    vit_large,
    vit_small,
)

__all__ = [
    # vision transformer
    "DinoVisionTransformer",
    "vit_base",
    "vit_giant2",
    "vit_large",
    "vit_small",
    # loader
    "DinoV2Loader",
]
