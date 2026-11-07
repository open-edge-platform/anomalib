# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Teacher feature extractor used by L2BT."""

from __future__ import annotations

import torch

from anomalib.models.components.feature_extractors import TimmFeatureExtractor


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

        self.layers = list(layers)
        # DINOv2-reg ViT-Base/14 loaded via timm. Patch tokens (CLS/register removed) from the two
        # selected blocks, with the backbone's final norm applied (matches get_intermediate_layers).
        self.fe = TimmFeatureExtractor(
            backbone="vit_base_patch14_reg4_dinov2",
            layers=[f"blocks.{i}" for i in self.layers],
            pre_trained=True,
            requires_grad=False,
            output_fmt="NLC",
            return_class_token=False,
            norm=True,
            dynamic_img_size=True,
        )

        self.patch_size = self.fe.patch_size
        self.embed_dim = self.fe.out_dims[0]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return intermediate patch features from the selected transformer layers.

        Args:
            x: Input image batch.

        Returns:
            Tuple containing the selected intermediate feature tensors.
        """
        features = self.fe(x)
        return features[f"blocks.{self.layers[0]}"], features[f"blocks.{self.layers[1]}"]
