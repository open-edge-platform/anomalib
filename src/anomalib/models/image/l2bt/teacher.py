# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Teacher feature extractor used by L2BT."""

from __future__ import annotations

import torch


class FeatureExtractor(torch.nn.Module):
    """Frozen DINOv2 teacher used to extract patch-level features."""

    def __init__(self, layers: list[int] | tuple[int, int]) -> None:
        """Initialize the teacher feature extractor.

        Args:
            layers: Indices of the transformer layers to extract.
        """
        super().__init__()

        self.fe = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")  # nosec B614
        self.layers = list(layers)

        self.patch_size = self.fe.patch_size
        self.embed_dim = self.fe.embed_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return intermediate patch features from the selected transformer layers.

        Args:
            x: Input image batch.

        Returns:
            Tuple containing the selected intermediate feature tensors.
        """
        middle_patch, last_patch = self.fe.get_intermediate_layers(x, self.layers)
        return middle_patch, last_patch

    def forward_last(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final normalized patch tokens from the teacher.

        Args:
            x: Input image batch.

        Returns:
            Final patch-token features reshaped to ``(num_patches, embed_dim)``.
        """
        return self.fe.forward_features(x)["x_norm_patchtokens"].view(-1, self.embed_dim)
