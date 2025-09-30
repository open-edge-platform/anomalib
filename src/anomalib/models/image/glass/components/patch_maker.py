# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Handles patch-based processing of feature maps."""

import torch


class PatchMaker:
    """Handles patch-based processing of feature maps.

    This class provides utilities for converting feature maps into patches,
    reshaping patch scores back to original dimensions, and computing global
    anomaly scores from patch-wise predictions.

    Attributes:
        patchsize (int): Size of each patch (patchsize x patchsize).
        stride (int or None): Stride used for patch extraction. Defaults to patchsize if None.
    """

    def __init__(self, patchsize: int, stride: int | None = None) -> None:
        self.patchsize = patchsize
        self.stride = stride if stride is not None else patchsize

    def patchify(
        self,
        features: torch.Tensor,
        return_spatial_info: bool = False,
    ) -> tuple[torch.Tensor, list[int]] | torch.Tensor:
        """Converts a batch of feature maps into patches.

        Args:
            features (torch.Tensor): Input feature maps of shape (B, C, H, W).
            return_spatial_info (bool): If True, also returns spatial patch count. Default is False.

        Returns:
            torch.Tensor: Output tensor of shape (B, N, C, patchsize, patchsize), where N is number of patches.
            list[int], optional: Number of patches in (height, width) dimensions, only if return_spatial_info is True.
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize,
            stride=self.stride,
            padding=padding,
            dilation=1,
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2],
            self.patchsize,
            self.patchsize,
            -1,
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    @staticmethod
    def unpatch_scores(x: torch.Tensor, batchsize: int) -> torch.Tensor:
        """Reshapes patch scores back into per-batch format.

        Args:
            x (torch.Tensor): Input tensor of shape (B * N, ...).
            batchsize (int): Original batch size.

        Returns:
            torch.Tensor: Reshaped tensor of shape (B, N, ...).
        """
        return x.reshape(batchsize, -1, *x.shape[1:])

    @staticmethod
    def compute_score(x: torch.Tensor) -> torch.Tensor:
        """Computes final anomaly scores from patch-wise predictions.

        Args:
            x (torch.Tensor): Patch scores of shape (B, N, 1).

        Returns:
            torch.Tensor: Final anomaly score per image, shape (B,).
        """
        x = x[:, :, 0]  # remove last dimension if singleton
        return torch.max(x, dim=1).values
