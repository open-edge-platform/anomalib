# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (C) 2025 Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Patch embedding module for DINOv2 Vision Transformers.

This module converts an image into a grid of patch embeddings using a strided
convolution. It supports square or rectangular image sizes and patch sizes,
optional output reshaping, and optional normalization.
"""

from collections.abc import Callable

import torch
from torch import nn


def make_2tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    """Ensure a value is represented as a 2-tuple.

    Args:
        x: Integer or tuple representing height/width.

    Returns:
        A tuple ``(h, w)``.
    """
    if isinstance(x, tuple):
        assert len(x) == 2, f"Expected tuple of length 2, got {len(x)}"
        return x
    return (x, x)


class PatchEmbed(nn.Module):
    """Image-to-patch embedding layer.

    Converts a 2D image tensor of shape ``(B, C, H, W)`` into a sequence of
    flattened patch embeddings of shape ``(B, N, D)`` where ``N`` is the number
    of patches and ``D`` is the embedding dimension.

    Args:
        img_size: Input image size (integer or ``(H, W)``).
        patch_size: Patch dimensions (integer or ``(H_p, W_p)``).
        in_chans: Number of input channels.
        embed_dim: Output embedding dimension.
        norm_layer: Optional normalization layer constructor.
        flatten_embedding: Whether to flatten to ``(B, N, D)`` (True) or return
            ``(B, H_p, W_p, D)`` (False).
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable[[int], nn.Module] | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_hw = make_2tuple(img_size)
        patch_hw = make_2tuple(patch_size)

        grid_h = image_hw[0] // patch_hw[0]
        grid_w = image_hw[1] // patch_hw[1]

        self.img_size = image_hw
        self.patch_size = patch_hw
        self.patches_resolution = (grid_h, grid_w)
        self.num_patches = grid_h * grid_w

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_hw, stride=patch_hw)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed the input image into patch tokens."""
        _, _, h, w = x.shape
        patch_h, patch_w = self.patch_size

        if h % patch_h != 0:
            msg = f"Input image height {h} must be divisible by patch height {patch_h}"
            raise ValueError(msg)
        if w % patch_w != 0:
            msg = f"Input image width {w} must be divisible by patch width {patch_w}"
            raise ValueError(msg)

        x = self.proj(x)  # (B, D, H', W')
        h_out, w_out = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = self.norm(x)

        if not self.flatten_embedding:
            x = x.reshape(-1, h_out, w_out, self.embed_dim)  # (B, H', W', D)

        return x

    def flops(self) -> float:
        """Compute FLOPs for the patch embedding layer."""
        grid_h, grid_w = self.patches_resolution
        flops = grid_h * grid_w * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        flops += grid_h * grid_w * self.embed_dim  # normalization cost
        return float(flops)
