# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Rescale and smooth patch-level anomaly scores into segmentation masks."""

import numpy as np
import torch
import torch.nn.functional as f

try:
    import kornia.filters as kf
except ImportError as e:
    msg = "kornia is required for RescaleSegmentor. Install with: pip install kornia"
    raise ImportError(msg) from e


class RescaleSegmentor:
    """Rescales patch-level scores to full-resolution smoothed segmentation masks.

    Args:
        target_size: Output spatial size (H, W) for segmentation maps.
        kernel_size: Gaussian blur kernel size (must be odd). Defaults to 33.
        sigma: Gaussian blur sigma. Defaults to 4.0.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (288, 288),
        kernel_size: int = 33,
        sigma: float = 4.0,
    ) -> None:
        if kernel_size < 1 or kernel_size % 2 == 0:
            msg = f"kernel_size must be a positive odd integer, got {kernel_size}"
            raise ValueError(msg)
        self.target_size = target_size
        self.kernel_size = kernel_size
        self.sigma = sigma

    def convert_to_segmentation(
        self,
        patch_scores: np.ndarray | torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Convert patch scores to smoothed segmentation masks.

        Upsamples via bilinear interpolation then applies Gaussian smoothing.

        Args:
            patch_scores: Patch-wise scores of shape (N, H_patch, W_patch).
            device: Device for computation. Inferred from input if None.

        Returns:
            torch.Tensor: Smoothed segmentation masks of shape (N, H, W).
        """
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)

            if device is not None:
                patch_scores = patch_scores.to(device)

            scores = patch_scores.unsqueeze(1)
            scores = f.interpolate(
                scores,
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            )

            scores = kf.gaussian_blur2d(
                scores,
                kernel_size=(self.kernel_size, self.kernel_size),
                sigma=(self.sigma, self.sigma),
                border_type="reflect",
            )

        return scores.squeeze(1)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"{self.__class__.__name__}(target_size={self.target_size}, "
            f"kernel_size={self.kernel_size}, sigma={self.sigma})"
        )
