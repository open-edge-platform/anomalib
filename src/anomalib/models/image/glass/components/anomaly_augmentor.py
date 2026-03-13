# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GLASS-specific anomaly augmentation module.

This module implements the anomaly synthesis strategy from the GLASS paper,
generating both local anomaly images and corresponding feature-level masks.
It follows the official GLASS implementation closely, including:
    - Dual Perlin noise mask generation with union/intersection/single modes
    - max_pool2d downsampling for feature-level masks
    - Normal-distributed blend factor
    - Matching augmentation transforms

Paper: `A Unified Anomaly Synthesis Strategy with Gradient Ascent for Industrial
    Anomaly Detection and Localization <https://arxiv.org/pdf/2407.09359>`
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as f
from torch import nn
from torchvision import io
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import v2

from anomalib.data.utils.generators.perlin import generate_perlin_noise

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GlassAnomalyAugmentor(nn.Module):
    """GLASS-specific anomaly augmentation module.

    Generates augmented images and feature-level masks matching the official
    GLASS implementation. For each input image, it:
        1. Loads a random DTD texture source image
        2. Applies random augmentations (3 of 9 transforms)
        3. Generates dual Perlin noise masks (union/intersection/single)
        4. Blends the augmented source with the input using the large mask
        5. Downsamples the mask via max_pool2d for feature-level supervision

    Args:
        anomaly_source_path: Path to directory containing anomaly source images
            (e.g., DTD texture dataset).
        input_size: Input image size after center crop (height = width).
            Defaults to ``288``.
        downsampling: Downsampling factor for feature-level masks. The feature
            map size is ``input_size // downsampling``. Defaults to ``8``.
        blend_mean: Mean of the normal distribution for blend factor beta.
            Defaults to ``0.5``.
        blend_std: Standard deviation of the normal distribution for blend
            factor beta. Defaults to ``0.1``.
        perlin_scale_min: Minimum Perlin noise scale exponent. Defaults to ``0``.
        perlin_scale_max: Maximum Perlin noise scale exponent. Defaults to ``6``.

    Example:
        >>> augmentor = GlassAnomalyAugmentor(
        ...     anomaly_source_path="/path/to/dtd/images",
        ...     input_size=288,
        ... )
        >>> images = torch.randn(4, 3, 288, 288)  # normalized batch
        >>> aug_images, mask_s = augmentor(images)
        >>> print(aug_images.shape)  # [4, 3, 288, 288]
        >>> print(mask_s.shape)  # [4, 1, 36, 36] (288//8 = 36)
    """

    def __init__(
        self,
        anomaly_source_path: Path | str | None = None,
        input_size: int = 288,
        downsampling: int = 8,
        blend_mean: float = 0.5,
        blend_std: float = 0.1,
        perlin_scale_min: int = 0,
        perlin_scale_max: int = 6,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.downsampling = downsampling
        self.feat_size = input_size // downsampling
        self.blend_mean = blend_mean
        self.blend_std = blend_std
        self.perlin_scale_min = perlin_scale_min
        self.perlin_scale_max = perlin_scale_max

        self.anomaly_source_paths: list[Path] = []
        if anomaly_source_path is not None:
            for img_ext in IMG_EXTENSIONS:
                self.anomaly_source_paths.extend(
                    Path(anomaly_source_path).rglob("*" + img_ext),
                )

        # Augmenters matching the official GLASS implementation
        self.augmenters = [
            v2.ColorJitter(contrast=(0.8, 1.2)),
            v2.ColorJitter(brightness=(0.8, 1.2)),
            v2.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            v2.RandomHorizontalFlip(p=1),
            v2.RandomVerticalFlip(p=1),
            v2.RandomGrayscale(p=1),
            v2.RandomAutocontrast(p=1),
            v2.RandomEqualize(p=1),
            v2.RandomAffine(degrees=(-45, 45)),
        ]

    def _rand_augmenter(self) -> v2.Compose:
        """Create a random augmentation pipeline matching the official GLASS.

        Picks 3 random augmentations from the list and wraps them with resize
        and normalization. Operates on tensors.

        Returns:
            v2.Compose: Composed augmentation pipeline.
        """
        aug_idx = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        return v2.Compose([
            v2.Resize(self.input_size, antialias=True),
            self.augmenters[aug_idx[0]],
            self.augmenters[aug_idx[1]],
            self.augmenters[aug_idx[2]],
            v2.CenterCrop(self.input_size),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _generate_perlin_threshold(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate a binary threshold mask from Perlin noise.

        Uses anomalib's ``generate_perlin_noise`` and applies random rotation
        via affine transform and thresholds at 0.5.

        Args:
            height: Height of the noise pattern.
            width: Width of the noise pattern.
            device: Device to generate noise on.

        Returns:
            torch.Tensor: Binary mask of shape ``[H, W]`` with values 0 or 1.
        """
        # Random scale matching official: 2^randint(min, max)
        scalex = 2 ** int(np.random.randint(self.perlin_scale_min, self.perlin_scale_max))
        scaley = 2 ** int(np.random.randint(self.perlin_scale_min, self.perlin_scale_max))

        perlin_noise = generate_perlin_noise(height, width, scale=(scalex, scaley), device=device)

        # Rotate the noise pattern (matching official imgaug affine rotate)
        perlin_noise = perlin_noise.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        angle = torch.empty(1).uniform_(-90, 90).item()
        perlin_noise = v2.functional.affine(
            perlin_noise,
            angle=angle,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0],
            interpolation=v2.InterpolationMode.BILINEAR,
            fill=[0.0],
        )
        perlin_noise = perlin_noise.squeeze(0).squeeze(0)  # [H, W]

        return torch.where(
            perlin_noise > 0.5,
            torch.ones_like(perlin_noise),
            torch.zeros_like(perlin_noise),
        )

    def _generate_perlin_mask(
        self,
        img_height: int,
        img_width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate dual Perlin noise masks for anomaly synthesis.

        Creates a binary mask by combining two Perlin noise patterns using
        union, intersection, or single mode (each with 1/3 probability).
        Returns both the downsampled feature-level mask (via max_pool2d)
        and the full-resolution mask.

        Args:
            img_height: Height of the input image.
            img_width: Width of the input image.
            device: Device to generate masks on.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - mask_s: Downsampled mask of shape ``[feat_size, feat_size]``
                - mask_l: Full-resolution mask of shape ``[H, W]``
        """
        feat_size = self.feat_size
        mask_s = torch.zeros((feat_size, feat_size), device=device)
        mask_l = torch.zeros((img_height, img_width), device=device)

        # Keep generating until we get a non-empty mask
        while mask_s.max() == 0:
            perlin_thr_1 = self._generate_perlin_threshold(img_height, img_width, device)
            perlin_thr_2 = self._generate_perlin_threshold(img_height, img_width, device)

            temp = torch.rand(1).item()
            if temp > 2 / 3:
                # Union
                perlin_thr = perlin_thr_1 + perlin_thr_2
                perlin_thr = torch.where(
                    perlin_thr > 0,
                    torch.ones_like(perlin_thr),
                    torch.zeros_like(perlin_thr),
                )
            elif temp > 1 / 3:
                # Intersection
                perlin_thr = perlin_thr_1 * perlin_thr_2
            else:
                # Single
                perlin_thr = perlin_thr_1

            # Downsample via max_pool2d (preserves any anomaly in receptive field)
            down_ratio_y = img_height // feat_size
            down_ratio_x = img_width // feat_size
            mask_l = perlin_thr
            mask_s = (
                f
                .max_pool2d(
                    perlin_thr.unsqueeze(0).unsqueeze(0),
                    (down_ratio_y, down_ratio_x),
                )
                .squeeze(0)
                .squeeze(0)
            )

        return mask_s, mask_l

    def _augment_single(
        self,
        img: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate augmented image and feature-level mask for a single image.

        Loads a random DTD source image as a tensor and applies the augmentation
        pipeline (Resize -> 3 random augments -> CenterCrop -> Normalize).

        Args:
            img: Normalized input image of shape ``[C, H, W]``.
            device: Device for computation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - aug_image: Blended augmented image ``[C, H, W]``
                - mask_s: Feature-level mask ``[1, feat_size, feat_size]``
        """
        _, h, w = img.shape

        # Load random anomaly source as tensor
        if self.anomaly_source_paths:
            source_idx = int(torch.randint(len(self.anomaly_source_paths), (1,)).item())
            source_path = self.anomaly_source_paths[source_idx]
            source_img = io.read_image(str(source_path), mode=io.ImageReadMode.RGB).float().to(device) / 255.0
        else:
            # Fallback: create a random noise tensor
            source_img = torch.rand(3, h, w, device=device)

        # Apply the full augmentation pipeline on tensor:
        # Resize -> 3 random augments -> CenterCrop -> Normalize
        transform = self._rand_augmenter()
        aug_source = transform(source_img)  # [C, H, W]

        # Generate dual Perlin mask
        mask_s, mask_l = self._generate_perlin_mask(h, w, device)

        # Blend: img * (1 - mask_l) + (1 - beta) * aug * mask_l + beta * img * mask_l
        beta = np.random.normal(loc=self.blend_mean, scale=self.blend_std)
        beta = float(np.clip(beta, 0.2, 0.8))

        mask_l_3d = mask_l.unsqueeze(0)  # [1, H, W]
        aug_image = img * (1 - mask_l_3d) + (1 - beta) * aug_source * mask_l_3d + beta * img * mask_l_3d

        return aug_image, mask_s.unsqueeze(0)  # [C,H,W], [1, feat_h, feat_w]

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate augmented images and feature-level masks for a batch.

        Args:
            images: Batch of normalized images of shape ``[B, C, H, W]``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - aug_images: Batch of augmented images ``[B, C, H, W]``
                - masks_s: Batch of feature-level masks ``[B, 1, feat_h, feat_w]``
        """
        device = images.device
        batch_aug = []
        batch_mask = []

        for i in range(images.shape[0]):
            aug_img, mask_s = self._augment_single(images[i], device)
            batch_aug.append(aug_img)
            batch_mask.append(mask_s)

        return torch.stack(batch_aug), torch.stack(batch_mask)
