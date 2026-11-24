# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Supervised auxiliary-domain loss for the UniADet head."""

import torch
from torch import nn
from torch.nn import functional

from .head import UniADetLogits


class UniADetLoss(nn.Module):
    """Sum image cross-entropy, pixel focal loss, and two-class Dice loss across scales.

    Args:
        gamma (float): Focal-loss exponent. Defaults to ``2.0``.
        smooth (float): Positive Dice smoothing constant. Defaults to ``1.0``.
        focal_smoothing (float): Binary focal-label smoothing. Defaults to ``1e-5``.
    """

    def __init__(self, gamma: float = 2.0, smooth: float = 1.0, focal_smoothing: float = 1e-5) -> None:
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth
        self.focal_smoothing = focal_smoothing

    def forward(self, logits: UniADetLogits, labels: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary-domain supervision from real anomaly labels and masks.

        Args:
            logits (UniADetLogits): Per-layer image and pixel logits.
            labels (torch.Tensor): Binary image labels of shape ``(B,)``.
            masks (torch.Tensor): Binary anomaly masks of shape ``(B, H, W)``.

        Returns:
            torch.Tensor: Scalar differentiable loss, summed over layers.
        """
        masks = masks.long()
        targets = functional.one_hot(masks, num_classes=2).permute(0, 3, 1, 2).float()
        losses = []
        for image, pixel in zip(logits.image, logits.pixel, strict=True):
            probabilities = functional.interpolate(
                pixel.softmax(dim=1),
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            target_probability = probabilities.gather(1, masks.unsqueeze(1)).squeeze(1)
            target_probability = target_probability * (1 - 2 * self.focal_smoothing) + 2 * self.focal_smoothing
            focal = (-((1 - target_probability) ** self.gamma) * target_probability.log()).mean()
            intersection = (probabilities * targets).sum(dim=(-2, -1))
            denominator = probabilities.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
            dice = (1 - (2 * intersection + self.smooth) / (denominator + self.smooth)).mean(dim=0).sum()
            losses.append(functional.cross_entropy(image, labels.long()) + focal + dice)
        return torch.stack(losses).sum()
