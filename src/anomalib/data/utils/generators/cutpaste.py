# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CutPaste-based synthetic anomaly generator.

This module implements a lightweight CutPaste strategy to synthesize local defects.
A random patch is cropped from an input image, optionally transformed, and pasted to
another location using soft alpha blending.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torchvision.transforms import v2


class CutPasteGenerator(nn.Module):
    """Generate synthetic local anomalies via CutPaste.

    The generator samples a rectangular patch from the input image, applies lightweight
    geometric and photometric perturbations, and pastes it in a different location.
    Blending is applied only inside the pasted region.

    Args:
        mode (Literal["normal", "scar", "union"]): CutPaste variant.
            - ``"normal"`` samples regular rectangular patches.
            - ``"scar"`` samples elongated thin patches.
            - ``"union"`` randomly selects normal or scar each call.
            Defaults to ``"normal"``.
        probability (float): Probability of applying augmentation. If sampling fails,
            the input is returned unchanged. Defaults to ``0.5``.
        blend_factor (float): Alpha blending value in ``[0, 1]`` applied in the pasted
            region. Defaults to ``0.5``.
        patch_size_ratio (tuple[float, float]): Min/max patch side ratio relative to
            image dimensions. Defaults to ``(0.1, 0.3)``.
        rotation_range (tuple[float, float]): Rotation range (degrees) for the copied
            patch. Defaults to ``(-30.0, 30.0)``.
        scar_rotation_range (tuple[float, float]): Rotation range (degrees) used when
            mode is ``"scar"`` (or scar is selected in ``"union"``). Defaults to
            ``(-10.0, 10.0)``.
        brightness_shift_range (tuple[float, float]): Multiplicative brightness scaling
            applied to the copied patch. Defaults to ``(0.9, 1.1)``.
        enable_edge_blur (bool): Whether to apply slight Gaussian blur to copied
            patches to reduce hard cut boundaries. Defaults to ``True``.
        edge_blur_kernel_size (int): Kernel size used for Gaussian blur when
            ``enable_edge_blur=True``. Must be odd and >= 3. Defaults to ``3``.
        scar_length_range (tuple[int, int]): Pixel range for long side of scar patches.
            Defaults to ``(50, 150)``.
        scar_thickness_range (tuple[int, int]): Pixel range for short side of scar
            patches. Defaults to ``(5, 20)``.
        scratch_probability (float): Backward-compatible parameter from earlier
            implementation. It is ignored when ``mode`` is explicitly set and retained
            only for API compatibility.
        scratch_aspect_ratio_range (tuple[float, float]): Backward-compatible parameter
            from earlier implementation. Retained for API compatibility.
        enable_hflip (bool): Whether to randomly apply horizontal flip. Defaults to
            ``True``.
        enable_vflip (bool): Whether to randomly apply vertical flip. Defaults to
            ``False``.
        enable_color_jitter (bool): Whether to apply subtle brightness/contrast jitter.
            Defaults to ``True``.
        color_jitter_strength (float): Jitter strength around identity for brightness
            and contrast. Defaults to ``0.1``.
    """

    def __init__(
        self,
        mode: Literal["normal", "scar", "union"] = "normal",
        probability: float = 0.5,
        blend_factor: float = 0.5,
        patch_size_ratio: tuple[float, float] = (0.1, 0.3),
        rotation_range: tuple[float, float] = (-30.0, 30.0),
        scar_rotation_range: tuple[float, float] = (-10.0, 10.0),
        enable_hflip: bool = True,
        enable_vflip: bool = False,
        enable_color_jitter: bool = True,
        color_jitter_strength: float = 0.1,
        brightness_shift_range: tuple[float, float] = (0.9, 1.1),
        enable_edge_blur: bool = True,
        edge_blur_kernel_size: int = 3,
        scar_length_range: tuple[int, int] = (50, 150),
        scar_thickness_range: tuple[int, int] = (5, 20),
        scratch_probability: float = 0.3,
        scratch_aspect_ratio_range: tuple[float, float] = (3.0, 8.0),
    ) -> None:
        super().__init__()

        if mode not in {"normal", "scar", "union"}:
            msg = f"mode must be one of ['normal', 'scar', 'union'], got {mode}"
            raise ValueError(msg)
        if not 0.0 <= probability <= 1.0:
            msg = f"probability must be in [0, 1], got {probability}"
            raise ValueError(msg)
        if not 0.0 <= blend_factor <= 1.0:
            msg = f"blend_factor must be in [0, 1], got {blend_factor}"
            raise ValueError(msg)
        if (
            patch_size_ratio[0] <= 0
            or patch_size_ratio[1] <= 0
            or patch_size_ratio[0] > patch_size_ratio[1]
        ):
            msg = f"patch_size_ratio must satisfy 0 < min <= max, got {patch_size_ratio}"
            raise ValueError(msg)
        if color_jitter_strength < 0:
            msg = f"color_jitter_strength must be >= 0, got {color_jitter_strength}"
            raise ValueError(msg)
        if scar_rotation_range[0] > scar_rotation_range[1]:
            msg = f"scar_rotation_range must satisfy min <= max, got {scar_rotation_range}"
            raise ValueError(msg)
        if brightness_shift_range[0] <= 0 or brightness_shift_range[0] > brightness_shift_range[1]:
            msg = f"brightness_shift_range must satisfy 0 < min <= max, got {brightness_shift_range}"
            raise ValueError(msg)
        if enable_edge_blur and (edge_blur_kernel_size < 3 or edge_blur_kernel_size % 2 == 0):
            msg = f"edge_blur_kernel_size must be odd and >= 3, got {edge_blur_kernel_size}"
            raise ValueError(msg)
        if scar_length_range[0] <= 0 or scar_length_range[0] > scar_length_range[1]:
            msg = f"scar_length_range must satisfy 0 < min <= max, got {scar_length_range}"
            raise ValueError(msg)
        if scar_thickness_range[0] <= 0 or scar_thickness_range[0] > scar_thickness_range[1]:
            msg = f"scar_thickness_range must satisfy 0 < min <= max, got {scar_thickness_range}"
            raise ValueError(msg)
        if not 0.0 <= scratch_probability <= 1.0:
            msg = f"scratch_probability must be in [0, 1], got {scratch_probability}"
            raise ValueError(msg)
        if scratch_aspect_ratio_range[0] < 1.0 or scratch_aspect_ratio_range[0] > scratch_aspect_ratio_range[1]:
            msg = f"scratch_aspect_ratio_range must satisfy 1 <= min <= max, got {scratch_aspect_ratio_range}"
            raise ValueError(msg)

        self.mode = mode
        self.probability = probability
        self.blend_factor = blend_factor
        self.patch_size_ratio = patch_size_ratio
        self.rotation_range = rotation_range
        self.scar_rotation_range = scar_rotation_range
        self.enable_hflip = enable_hflip
        self.enable_vflip = enable_vflip
        self.enable_color_jitter = enable_color_jitter
        self.color_jitter_strength = color_jitter_strength
        self.brightness_shift_range = brightness_shift_range
        self.enable_edge_blur = enable_edge_blur
        self.edge_blur_kernel_size = edge_blur_kernel_size
        self.scar_length_range = scar_length_range
        self.scar_thickness_range = scar_thickness_range
        # Backward-compatible fields retained intentionally.
        self.scratch_probability = scratch_probability
        self.scratch_aspect_ratio_range = scratch_aspect_ratio_range

    def _sample_patch_normal(self, height: int, width: int, device: torch.device) -> tuple[int, int]:
        """Sample regular CutPaste patch dimensions."""
        min_ratio, max_ratio = self.patch_size_ratio
        min_h = max(1, int(min_ratio * height))
        max_h = max(2, int(max_ratio * height) + 1)
        min_w = max(1, int(min_ratio * width))
        max_w = max(2, int(max_ratio * width) + 1)

        patch_h = torch.randint(min_h, max_h, (1,), device=device).item()
        patch_w = torch.randint(min_w, max_w, (1,), device=device).item()
        return min(max(patch_h, 1), height), min(max(patch_w, 1), width)

    def _sample_patch_scar(self, height: int, width: int, device: torch.device) -> tuple[int, int]:
        """Sample scar-like elongated patch dimensions."""
        long_min = min(self.scar_length_range[0], max(height, width))
        long_max = min(self.scar_length_range[1], max(height, width))
        thin_min = min(self.scar_thickness_range[0], min(height, width))
        thin_max = min(self.scar_thickness_range[1], min(height, width))
        if long_min > long_max:
            long_min = long_max
        if thin_min > thin_max:
            thin_min = thin_max
        long_side = torch.randint(long_min, long_max + 1, (1,), device=device).item()
        thin_side = torch.randint(thin_min, thin_max + 1, (1,), device=device).item()

        if torch.rand(1, device=device).item() < 0.5:
            patch_h, patch_w = thin_side, long_side
        else:
            patch_h, patch_w = long_side, thin_side
        return min(max(patch_h, 1), height), min(max(patch_w, 1), width)

    def _sample_patch_size(self, height: int, width: int, device: torch.device) -> tuple[int, int, str]:
        """Sample patch size according to selected CutPaste mode."""
        selected_mode = self.mode
        if selected_mode == "union":
            selected_mode = "normal" if torch.rand(1, device=device).item() < 0.5 else "scar"
        if selected_mode == "normal":
            patch_h, patch_w = self._sample_patch_normal(height, width, device)
            return patch_h, patch_w, selected_mode
        patch_h, patch_w = self._sample_patch_scar(height, width, device)
        return patch_h, patch_w, selected_mode

    def _sample_rotation_angle(self, selected_mode: str, device: torch.device) -> float:
        """Sample rotation angle with scar-specific range support."""
        min_angle, max_angle = self.rotation_range
        if selected_mode == "scar":
            min_angle, max_angle = self.scar_rotation_range
        return torch.rand(1, device=device).item() * (max_angle - min_angle) + min_angle

    def _transform_patch(self, patch: torch.Tensor, selected_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply lightweight transformations and return transformed patch + valid mask.

        Args:
            patch: Patch tensor of shape ``(C, H, W)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Transformed patch ``(C, H, W)`` and a
            valid-pixel mask ``(1, H, W)``.
        """
        device = patch.device
        transformed = patch
        valid_mask = torch.ones_like(patch[:1])

        angle = self._sample_rotation_angle(selected_mode=selected_mode, device=device)
        fill_value = float(patch.mean().item())
        transformed = v2.functional.rotate(
            transformed,
            angle=angle,
            interpolation=v2.InterpolationMode.BILINEAR,
            expand=False,
            fill=fill_value,
        )
        valid_mask = v2.functional.rotate(
            valid_mask,
            angle=angle,
            interpolation=v2.InterpolationMode.NEAREST,
            expand=False,
            fill=0.0,
        )

        if self.enable_hflip and torch.rand(1, device=device).item() < 0.5:
            transformed = v2.functional.horizontal_flip(transformed)
            valid_mask = v2.functional.horizontal_flip(valid_mask)
        if self.enable_vflip and torch.rand(1, device=device).item() < 0.5:
            transformed = v2.functional.vertical_flip(transformed)
            valid_mask = v2.functional.vertical_flip(valid_mask)

        # Explicit multiplicative brightness shift improves defect visibility realism.
        brightness_scale = self.brightness_shift_range[0] + (
            (self.brightness_shift_range[1] - self.brightness_shift_range[0]) * torch.rand(1, device=device).item()
        )
        transformed = transformed * brightness_scale

        if self.enable_edge_blur:
            transformed = v2.functional.gaussian_blur(
                transformed,
                kernel_size=[self.edge_blur_kernel_size, self.edge_blur_kernel_size],
            )

        if self.enable_color_jitter and self.color_jitter_strength > 0:
            brightness_factor = 1.0 + (
                (torch.rand(1, device=device).item() * 2 - 1) * self.color_jitter_strength
            )
            contrast_factor = 1.0 + (
                (torch.rand(1, device=device).item() * 2 - 1) * self.color_jitter_strength
            )
            transformed = v2.functional.adjust_brightness(transformed, brightness_factor)
            transformed = v2.functional.adjust_contrast(transformed, contrast_factor)

        transformed = transformed * valid_mask
        transformed = transformed.clamp(0.0, 1.0) if transformed.is_floating_point() else transformed
        return transformed, valid_mask

    def generate(self, image: torch.Tensor) -> torch.Tensor:
        """Generate a CutPaste anomaly from a single image.

        Args:
            image (torch.Tensor): Input image tensor with shape ``(C, H, W)``.

        Returns:
            torch.Tensor: Output image tensor with shape ``(C, H, W)``.
        """
        if image.ndim != 3:
            msg = f"Expected image shape (C, H, W), got {tuple(image.shape)}"
            raise ValueError(msg)

        device = image.device
        _, height, width = image.shape
        if torch.rand(1, device=device).item() > self.probability:
            return image

        patch_h, patch_w, selected_mode = self._sample_patch_size(height, width, device)

        if patch_h == height and patch_w == width:
            return image

        src_y = torch.randint(0, height - patch_h + 1, (1,), device=device).item()
        src_x = torch.randint(0, width - patch_w + 1, (1,), device=device).item()

        patch = image[:, src_y : src_y + patch_h, src_x : src_x + patch_w].clone()
        patch, patch_valid_mask = self._transform_patch(patch, selected_mode=selected_mode)

        max_attempts = 20
        dst_y, dst_x = src_y, src_x
        for _ in range(max_attempts):
            cand_y = torch.randint(0, height - patch_h + 1, (1,), device=device).item()
            cand_x = torch.randint(0, width - patch_w + 1, (1,), device=device).item()
            if cand_y != src_y or cand_x != src_x:
                dst_y, dst_x = cand_y, cand_x
                break
        else:
            return image

        output = image.clone()
        region = output[:, dst_y : dst_y + patch_h, dst_x : dst_x + patch_w]
        blended_region = (
            (1.0 - self.blend_factor) * region + self.blend_factor * patch
        )
        valid = patch_valid_mask.expand_as(region) > 0.5
        region[valid] = blended_region[valid]
        output[:, dst_y : dst_y + patch_h, dst_x : dst_x + patch_w] = region

        return output

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Alias for ``generate`` to support module-style usage."""
        return self.generate(image)
