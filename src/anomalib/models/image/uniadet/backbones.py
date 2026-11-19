# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Frozen vision encoders for UniADet, using timm rather than vendored source."""

from typing import NamedTuple

import timm
import torch
from torch import nn
from torch.nn import functional


class BackboneSpec(NamedTuple):
    """Pretrained encoder name, input resolution, patch size, and feature dimension."""

    name: str
    image_size: int
    patch_size: int
    feature_dim: int


BACKBONES = {
    "clip": BackboneSpec("vit_large_patch14_clip_quickgelu_336.openai", 518, 14, 768),
    "dinov2": BackboneSpec("vit_large_patch14_reg4_dinov2.lvd142m", 518, 14, 1024),
    "dinov3": BackboneSpec("vit_large_patch16_dinov3.lvd1689m", 512, 16, 1024),
}


class HierarchicalFeatures(NamedTuple):
    """Image and patch tokens, ordered by increasing transformer block index."""

    image_tokens: list[torch.Tensor]
    patch_tokens: list[torch.Tensor]


class UniADetBackbone(nn.Module):
    """Extract frozen hierarchical features from a pretrained vision transformer.

    CLIP uses the original QuickGELU activation and projects both image and patch
    tokens into its 768-dimensional embedding space. DINO register tokens are
    excluded from the patch grid. Spatial resolution is fixed at construction;
    batch size remains dynamic.

    Args:
        backbone (str): One of ``"clip"``, ``"dinov2"`` or ``"dinov3"``.
        layers (tuple[int, ...]): One-based transformer block numbers.
        pretrained (bool): Download pretrained backbone weights.
        image_size (int | None): Square input resolution. Defaults to the paper's resolution.
        protocol (str): ``"legacy"`` preserves existing features; ``"author"`` matches
            the published detection forwards and positional-embedding conventions.

    Raises:
        ValueError: If the backbone, block numbers, or image size are invalid.
    """

    def __init__(
        self,
        backbone: str = "dinov2",
        layers: tuple[int, ...] = (12, 15, 18, 21, 24),
        pretrained: bool = True,
        image_size: int | None = None,
        protocol: str = "legacy",
    ) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            msg = f"Unknown UniADet backbone {backbone!r}; choose from {tuple(BACKBONES)}."
            raise ValueError(msg)
        if protocol not in {"legacy", "author"}:
            msg = "protocol must be 'legacy' or 'author'."
            raise ValueError(msg)
        if not layers or tuple(sorted(set(layers))) != layers or not all(1 <= layer <= 24 for layer in layers):
            msg = "layers must contain unique, increasing, one-based block numbers between 1 and 24."
            raise ValueError(msg)
        spec = BACKBONES[backbone]
        self.image_size = spec.image_size if image_size is None else image_size
        if self.image_size <= 0 or self.image_size % spec.patch_size:
            msg = f"image_size must be a positive multiple of {spec.patch_size}."
            raise ValueError(msg)
        self.feature_dim = spec.feature_dim
        self.grid_size = self.image_size // spec.patch_size
        self.indices = [layer - 1 for layer in layers]
        self.is_clip = backbone == "clip"
        self.normalize_intermediates = protocol == "legacy" or self.is_clip
        self.encoder = timm.create_model(
            spec.name,
            pretrained=pretrained,
            img_size=336 if protocol == "author" and self.is_clip else self.image_size,
            dynamic_img_size=False,
            global_pool="token",
        )
        if protocol == "author":
            if self.is_clip:
                self._resize_clip_positions()
            elif backbone == "dinov3":
                # Official checkpoints store these periods rounded to bfloat16.
                self.encoder.rope.periods.copy_(self.encoder.rope.periods.bfloat16().float())
                self.encoder.rope.pos_embed_cached = None
        self.encoder.requires_grad_(requires_grad=False)
        self.encoder.eval()

    def _resize_clip_positions(self) -> None:
        positions = self.encoder.pos_embed.detach()
        old_grid = self.encoder.patch_embed.grid_size[0]
        self.encoder.patch_embed.set_input_size(img_size=self.image_size)
        if old_grid != self.grid_size:
            patches = positions[:, 1:].reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
            patches = functional.interpolate(
                patches,
                scale_factor=(self.grid_size + 0.1) / old_grid,
                mode="bicubic",
                align_corners=False,
                antialias=False,
            )
            patches = patches.flatten(2).transpose(1, 2)
            self.encoder.pos_embed = nn.Parameter(torch.cat((positions[:, :1], patches), dim=1), requires_grad=False)

    def train(self, mode: bool = True) -> "UniADetBackbone":
        """Keep the frozen encoder in evaluation mode, including during head training.

        Args:
            mode (bool): Requested training mode (ignored).

        Returns:
            UniADetBackbone: This encoder in evaluation mode.
        """
        del mode
        super().train(mode=False)
        return self

    def forward(self, images: torch.Tensor) -> HierarchicalFeatures:
        """Extract class tokens and patch grids without retaining a gradient graph.

        Args:
            images (torch.Tensor): Normalized images of shape ``(B, 3, H, W)``.

        Returns:
            HierarchicalFeatures: Image tokens ``(B, D)`` and patches ``(B, N, D)`` per layer.
        """
        with torch.no_grad():
            intermediates = self.encoder.forward_intermediates(
                images.to(dtype=self.encoder.patch_embed.proj.weight.dtype),
                indices=self.indices,
                return_prefix_tokens=True,
                norm=self.normalize_intermediates,
                output_fmt="NLC",
                intermediates_only=True,
            )
            image_tokens, patch_tokens = [], []
            for patches, prefixes in intermediates:
                image = prefixes[:, 0]
                if self.is_clip:
                    image = self.encoder.head(image)
                    projected_patches = self.encoder.head(patches)
                else:
                    projected_patches = patches
                image_tokens.append(image)
                patch_tokens.append(projected_patches)
        return HierarchicalFeatures(image_tokens, patch_tokens)
