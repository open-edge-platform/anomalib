# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch feature embedding pipeline for MH-PatchCore."""

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components import TimmFeatureExtractor

_PATCH_SIZE = 3
_PATCH_STRIDE = 1
_FEATURE_DIMENSION = 1024


class MHPatchcoreModel(nn.Module):
    """Extract ordered patch embeddings for MH-PatchCore.

    Args:
        backbone (str): Name of the timm backbone. Defaults to
            ``"wide_resnet50_2.tv2_in1k"``.
        layers (Sequence[str]): Ordered backbone layers used for feature
            extraction. Defaults to ``("layer2", "layer3")``.
        pre_trained (bool): Whether to load pretrained backbone weights.
            Defaults to ``True``.

    Raises:
        ValueError: If ``layers`` is empty.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2.tv2_in1k",
        layers: Sequence[str] = ("layer2", "layer3"),
        pre_trained: bool = True,
    ) -> None:
        super().__init__()
        if not layers:
            msg = "layers must contain at least one feature layer."
            raise ValueError(msg)

        self.backbone = backbone
        self.layers = tuple(layers)
        self.feature_extractor = TimmFeatureExtractor(
            backbone=backbone,
            layers=self.layers,
            pre_trained=pre_trained,
        ).eval()
        self._unfold = nn.Unfold(
            kernel_size=_PATCH_SIZE,
            stride=_PATCH_STRIDE,
            padding=_PATCH_SIZE // 2,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Extract MH-PatchCore embeddings from an image batch.

        Args:
            input_tensor (torch.Tensor): Image batch with shape ``[B, C, H, W]``.

        Returns:
            torch.Tensor: Patch embeddings with shape
                ``[B * H_ref * W_ref, 1024]``.
        """
        features = self.feature_extractor(input_tensor)
        return self.generate_embedding(features)

    def generate_embedding(self, features: dict[str | int, torch.Tensor]) -> torch.Tensor:
        """Construct ordered embeddings from hierarchical feature maps.

        Args:
            features (dict[str | int, torch.Tensor]): Feature maps keyed by
                backbone layer name.

        Returns:
            torch.Tensor: Batch-major, row-major patch embeddings with 1024
                features per patch.
        """
        patched_features: list[torch.Tensor] = []
        patch_grids: list[tuple[int, int]] = []
        for layer in self.layers:
            patches, grid = self._patchify(features[layer])
            patched_features.append(patches)
            patch_grids.append(grid)

        reference_grid = patch_grids[0]
        aligned_features = [patched_features[0]]
        aligned_features.extend(
            self._align_patches(patches, source_grid, reference_grid)
            for patches, source_grid in zip(patched_features[1:], patch_grids[1:], strict=True)
        )

        aligned_features = [patches.reshape(-1, *patches.shape[-3:]) for patches in aligned_features]
        mapped_features = [self._map_features(patches) for patches in aligned_features]
        return self._aggregate_features(torch.stack(mapped_features, dim=1))

    def _patchify(self, features: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        batch_size, channels, height, width = features.shape
        patches = self._unfold(features)
        patches = patches.transpose(1, 2).reshape(batch_size, -1, channels, _PATCH_SIZE, _PATCH_SIZE)
        return patches, (height, width)

    @staticmethod
    def _align_patches(
        patches: torch.Tensor,
        source_grid: tuple[int, int],
        target_grid: tuple[int, int],
    ) -> torch.Tensor:
        if source_grid == target_grid:
            return patches

        batch_size, _, channels, patch_height, patch_width = patches.shape
        source_height, source_width = source_grid
        target_height, target_width = target_grid
        patches = patches.reshape(
            batch_size,
            source_height,
            source_width,
            channels,
            patch_height,
            patch_width,
        )
        patches = patches.permute(0, 3, 4, 5, 1, 2).reshape(-1, 1, source_height, source_width)
        patches = F.interpolate(
            patches,
            size=target_grid,
            mode="bilinear",
            align_corners=False,
        )
        patches = patches.reshape(
            batch_size,
            channels,
            patch_height,
            patch_width,
            target_height,
            target_width,
        )
        return patches.permute(0, 4, 5, 1, 2, 3).reshape(
            batch_size,
            target_height * target_width,
            channels,
            patch_height,
            patch_width,
        )

    @staticmethod
    def _map_features(patches: torch.Tensor) -> torch.Tensor:
        patches = patches.reshape(len(patches), 1, -1)
        return F.adaptive_avg_pool1d(patches, _FEATURE_DIMENSION).squeeze(1)

    @staticmethod
    def _aggregate_features(features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, _FEATURE_DIMENSION).squeeze(1)
