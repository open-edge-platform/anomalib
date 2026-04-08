# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Teacher feature extractor used by L2BT."""

from __future__ import annotations

import torch

from anomalib.models.components.dinov2 import DinoV2Loader


class FeatureExtractor(torch.nn.Module):
    """Frozen DINOv2 teacher used to extract patch-level features."""

    def __init__(self, layers: list[int] | tuple[int, int]) -> None:
        """Initialize the teacher feature extractor.

        Args:
            layers: Indices of exactly two transformer layers to extract.
                Must have length 2 because ``forward`` unpacks the result
                into ``(middle_patch, last_patch)``.

        Raises:
            ValueError: If ``layers`` does not contain exactly 2 indices.
        """
        super().__init__()

        if len(layers) != 2:
            msg = f"FeatureExtractor requires exactly 2 layer indices (middle, last), got {len(layers)}: {list(layers)}"
            raise ValueError(msg)

        loader = DinoV2Loader()
        self.fe = loader.load("dinov2reg_base_14")
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
