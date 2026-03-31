# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for CutPaste synthetic anomaly generator."""

import pytest
import torch

from anomalib.data.utils.generators import CutPasteGenerator


def test_cutpaste_output_shape_matches_input() -> None:
    """Ensure generator preserves input tensor shape."""
    image = torch.rand(3, 128, 128)
    generator = CutPasteGenerator(probability=1.0)
    output = generator.generate(image)
    assert output.shape == image.shape


def test_cutpaste_modifies_image_when_probability_one() -> None:
    """Ensure output differs from input when augmentation is always applied."""
    image = torch.rand(3, 128, 128)
    generator = CutPasteGenerator(probability=1.0, blend_factor=0.7)
    output = generator.generate(image)
    assert not torch.equal(output, image)


def test_cutpaste_returns_original_when_probability_zero() -> None:
    """Ensure generator returns input unchanged when probability is zero."""
    image = torch.rand(3, 128, 128)
    generator = CutPasteGenerator(probability=0.0)
    output = generator.generate(image)
    assert torch.equal(output, image)


def test_cutpaste_modifies_local_region() -> None:
    """Ensure only a subset of pixels is modified in typical operation."""
    image = torch.rand(3, 128, 128)
    generator = CutPasteGenerator(probability=1.0, patch_size_ratio=(0.1, 0.2), blend_factor=0.6)
    output = generator.generate(image)

    diff_map = (output - image).abs().sum(dim=0) > 1e-6
    changed_pixels = int(diff_map.sum())

    assert changed_pixels > 0
    assert changed_pixels < image.shape[1] * image.shape[2]


def test_cutpaste_invalid_mode_raises() -> None:
    """Ensure invalid mode values are rejected."""
    with pytest.raises(ValueError, match="mode"):
        CutPasteGenerator(mode="invalid")  # type: ignore[arg-type]


def test_cutpaste_scar_mode_produces_elongated_patches() -> None:
    """Ensure scar mode samples elongated (high-aspect) patch sizes."""
    generator = CutPasteGenerator(mode="scar")
    aspects: list[float] = []
    for _ in range(50):
        h, w, _ = generator._sample_patch_size(height=256, width=256, device=torch.device("cpu"))  # noqa: SLF001
        aspects.append(max(h, w) / max(1, min(h, w)))
    assert max(aspects) >= 3.0


def test_cutpaste_union_mode_samples_both_variants() -> None:
    """Ensure union mode occasionally yields both regular and elongated patches."""
    generator = CutPasteGenerator(mode="union", patch_size_ratio=(0.2, 0.25))
    elongated = 0
    regular = 0
    for _ in range(120):
        h, w, _ = generator._sample_patch_size(height=256, width=256, device=torch.device("cpu"))  # noqa: SLF001
        aspect = max(h, w) / max(1, min(h, w))
        if aspect >= 3.0:
            elongated += 1
        else:
            regular += 1
    assert elongated > 0
    assert regular > 0


def test_cutpaste_brightness_shift_changes_patch_intensity() -> None:
    """Ensure multiplicative brightness shift alters patch intensities."""
    patch = torch.full((3, 64, 64), 0.5)
    generator = CutPasteGenerator(
        mode="normal",
        enable_hflip=False,
        enable_vflip=False,
        enable_color_jitter=False,
        rotation_range=(0.0, 0.0),
        brightness_shift_range=(1.2, 1.2),
    )
    transformed_patch, _ = generator._transform_patch(patch, selected_mode="normal")  # noqa: SLF001
    assert not torch.allclose(transformed_patch, patch)


def test_cutpaste_shape_safety_small_image() -> None:
    """Ensure no out-of-bounds issues on small images."""
    image = torch.rand(3, 32, 32)
    generator = CutPasteGenerator(
        mode="scar",
        probability=1.0,
        scar_length_range=(20, 150),
        scar_thickness_range=(2, 10),
    )
    output = generator.generate(image)
    assert output.shape == image.shape


def test_cutpaste_scar_rotation_range_is_used() -> None:
    """Ensure scar mode uses dedicated small rotation range."""
    generator = CutPasteGenerator(mode="scar", rotation_range=(-45.0, 45.0), scar_rotation_range=(-10.0, 10.0))
    angle = generator._sample_rotation_angle(selected_mode="scar", device=torch.device("cpu"))  # noqa: SLF001
    assert -10.0 <= angle <= 10.0
