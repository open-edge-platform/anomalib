# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""A utility class for rescaling and smoothing patch-level anomaly scores to generate segmentation masks."""

import kornia.filters as kf
import numpy as np
import torch
import torch.nn.functional as f


class RescaleSegmentor:
    """A utility class for rescaling and smoothing patch-level anomaly scores to generate segmentation masks.

    Attributes:
        target_size (tuple[int, int]): The spatial size (height and width) to which patch scores will be rescaled.
        smoothing (int): The standard deviation used for Gaussian smoothing.
    """

    def __init__(self, target_size: tuple[int, int] = (288, 288)) -> None:
        """Initializes the RescaleSegmentor.

        Args:
            target_size (tuple[int, int], optional): The desired output size (height, width)
                of segmentation maps. Defaults to ``(288, 288)``.
        """
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(
        self,
        patch_scores: np.ndarray | torch.Tensor,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Converts patch-level scores to smoothed segmentation masks.

        Upsamples patch scores to ``target_size`` via bilinear interpolation, then
        applies Gaussian smoothing via kornia (kernel_size=33, sigma=4, reflect
        padding) to match the official GLASS post-processing behaviour.

        Args:
            patch_scores (np.ndarray | torch.Tensor): Patch-wise scores of shape ``[N, H, W]``.
            device (torch.device): Device on which to perform computation.

        Returns:
            list[torch.Tensor]: A list of segmentation masks, each of shape ``[H, W]``,
                rescaled to ``target_size`` and smoothed.
        """
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)

            scores = patch_scores.to(device)
            scores = scores.unsqueeze(1)  # [N, 1, H, W]
            scores = f.interpolate(
                scores,
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            )

            scores = kf.gaussian_blur2d(
                scores,
                kernel_size=(33, 33),
                sigma=(self.smoothing, self.smoothing),
                border_type="reflect",
            )
            patch_scores = scores.squeeze(1)  # [N, H, W]

        return list(patch_scores)  # List of [H, W] tensors
