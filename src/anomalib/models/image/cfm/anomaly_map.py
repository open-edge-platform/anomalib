# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomaly map and score generation for CFM (Crossmodal Feature Mapping)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class CFMAnomalyMapGenerator(nn.Module):
    """Generates anomaly map and scores based on crossmodal discrepance.

    Computes the mapping error between 2d and 3d domains and applies a spatial filter.
    """

    def __init__(
        self,
        sigma: float = 4.0,
        topk_ratio: float = 0.001,
    ) -> None:
        """Initialize the maps generator.

        Args:
            sigma: Standard Deviation for Gaussian Blur.
            topk_ratio: Percentage of worst pixels used for the score of the image.
        """
        super().__init__()
        self.topk_ratio = topk_ratio

        # Substituted KNNGaussianBlur based on PIL with a 2D kernel PyTorch.
        kernel_size = int(2 * (int(sigma * 3) // 2) + 1)
        self.blur = GaussianBlurKernel(kernel_size=kernel_size, sigma=sigma)

    @staticmethod
    def compute_distance(real: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        """Compute normalized L2 distance between real and predicted features.

        Matches the original paper: ||normalize(pred) - normalize(real)||_2
        Input: (B, C, H, W).
        Output: (B, H, W).
        """
        real_norm = functional.normalize(real, dim=1)
        pred_norm = functional.normalize(predicted, dim=1)
        return (pred_norm - real_norm).pow(2).sum(dim=1).sqrt()

    def forward(
        self,
        rgb_feat: torch.Tensor,
        xyz_feat: torch.Tensor,
        pred_rgb: torch.Tensor,
        pred_xyz: torch.Tensor,
        target_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates combined map and global score.

        Returns:
            anomaly_map: Final map with shape (B, 1, H, W).
            pred_score: Global score of the image with shape (B,).
        """
        # 1. Distance 3D -> 2D
        dist_rgb = self.compute_distance(rgb_feat, pred_rgb)  # Shape: (B, H, W)

        # 2. Distance 2D -> 3D
        dist_xyz = self.compute_distance(xyz_feat, pred_xyz)  # Shape: (B, H, W)

        # 3. Combination via element-wise product (paper: cos_2d * cos_3d)
        combined_map = dist_rgb * dist_xyz

        # Aggiungiamo the dimension of the channel(B, 1, H, W)
        anomaly_map = combined_map.unsqueeze(1)

        # 4. Upsampling to original dimension of the image
        if anomaly_map.shape[-2:] != (target_size):
            anomaly_map = functional.interpolate(
                anomaly_map,
                size=(target_size),
                mode="bilinear",
                align_corners=False,
            )

        # 5. Smoothing
        anomaly_map = self.blur(anomaly_map)

        # 6. Computation of the score
        pred_score = self._compute_image_score(anomaly_map)

        return anomaly_map, pred_score

    def _compute_image_score(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        """Computes the global score evaluating the mean of the highest anomaly values(Top-K)."""
        b, _, h, w = anomaly_map.shape
        flat = anomaly_map.view(b, -1)

        # Computing how many pixels to consider based on topk_ratio
        k = max(1, int(h * w * self.topk_ratio))

        topk_values, _ = torch.topk(flat, k=k, dim=1)
        return topk_values.mean(dim=1)


class GaussianBlurKernel(nn.Module):
    """Spatial 2D Gaussian Filter."""

    def __init__(self, kernel_size: int, sigma: float) -> None:
        super().__init__()
        self.kernel_size = kernel_size

        # Grid Coordinates
        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2

        # 1D Gaussian Distribution
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()

        # Multiply to obtain the 2D kernel (1, 1, K, K)
        kernel_2d = g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)

        # register_buffer assures that the kernel goes on the GPU with the model
        # without being considered a trainable parameter by the optimizer.
        self.register_buffer("kernel", kernel_2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies a convolution on anomaly map."""
        padding = self.kernel_size // 2
        return functional.conv2d(x, self.kernel, padding=padding)
