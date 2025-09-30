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
        target_size (int): The spatial size (height and width) to which patch scores will be rescaled.
        smoothing (int): The standard deviation used for Gaussian smoothing.
    """

    def __init__(self, target_size: tuple[int, int] = (288, 288)) -> None:
        """Initializes the RescaleSegmentor.

        Args:
            target_size (int, optional): The desired output size (height/width) of segmentation maps. Defaults to 288.
        """
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(
        self,
        patch_scores: np.ndarray | torch.Tensor,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Converts patch-level scores to smoothed segmentation masks.

        Args:
            patch_scores (np.ndarray | torch.Tensor): Patch-wise scores of shape [N, H, W].
            device (torch.device): Device on which to perform computation.

        Returns:
            List[torch.Tensor]: A list of segmentation masks, each of shape [H, W],
                                rescaled to `target_size` and smoothed.
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
            patch_scores = scores.squeeze(1)  # [N, H, W]

        patch_stack = patch_scores.unsqueeze(1)  # [N, 1, H, W]
        smoothed_stack = kf.gaussian_blur2d(
            patch_stack,
            kernel_size=(33, 33),
            sigma=(self.smoothing, self.smoothing),
        )

        return [s.squeeze(0) for s in smoothed_stack]  # List of [H, W] tensors
