# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Independent implementation of UniADet's decoupled cosine classifiers (paper Eq. 8)."""

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional

from .backbones import HierarchicalFeatures


class UniADetLogits(NamedTuple):
    """Two-class image and pixel logits for every selected transformer block."""

    image: list[torch.Tensor]
    pixel: list[torch.Tensor]


class MSDCSHead(nn.Module):
    """Multi-scale decoupled classification and segmentation prototypes.

    Each layer has separate normal/anomaly prototypes for image and patch tokens.

    Args:
        feature_dim (int): Token dimension.
        num_layers (int): Number of selected transformer blocks.
        temperature (float): Positive cosine-logit temperature. Defaults to ``0.07``.

    Raises:
        ValueError: If a dimension or the temperature is nonpositive.
    """

    def __init__(self, feature_dim: int, num_layers: int, temperature: float = 0.07) -> None:
        super().__init__()
        if feature_dim <= 0 or num_layers <= 0 or temperature <= 0:
            msg = "feature_dim, num_layers, and temperature must be positive."
            raise ValueError(msg)
        self.temperature = temperature
        self.image_prototypes = nn.Parameter(torch.randn(num_layers, 2, feature_dim))
        self.pixel_prototypes = nn.Parameter(torch.randn(num_layers, 2, feature_dim))

    def forward(self, features: HierarchicalFeatures, grid_size: int) -> UniADetLogits:
        """Compute independent cosine logits for every scale.

        Args:
            features (HierarchicalFeatures): Image and patch tokens.
            grid_size (int): Number of patches along each spatial dimension.

        Returns:
            UniADetLogits: Image ``(B, 2)`` and pixel ``(B, 2, G, G)`` logits per scale.
        """
        image_weights = functional.normalize(self.image_prototypes, dim=-1)
        pixel_weights = functional.normalize(self.pixel_prototypes, dim=-1)
        image_logits, pixel_logits = [], []
        for layer, (image, patches) in enumerate(zip(features.image_tokens, features.patch_tokens, strict=True)):
            image_logits.append(functional.normalize(image, dim=-1) @ image_weights[layer].T / self.temperature)
            pixels = functional.normalize(patches, dim=-1) @ pixel_weights[layer].T / self.temperature
            pixel_logits.append(pixels.transpose(1, 2).reshape(-1, 2, grid_size, grid_size))
        return UniADetLogits(image_logits, pixel_logits)
