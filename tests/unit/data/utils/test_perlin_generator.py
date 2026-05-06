# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for PerlinAnomalyGenerator, focusing on dual_mask behavior."""

import torch

from anomalib.data.utils.generators.perlin import PerlinAnomalyGenerator, generate_perlin_noise


def test_perlin_noise_output_shape() -> None:
    """Verify noise has correct spatial dimensions."""
    noise = generate_perlin_noise(64, 128)
    assert noise.shape == (64, 128)


def test_perlin_noise_output_dtype() -> None:
    """Verify noise is float32."""
    noise = generate_perlin_noise(32, 32)
    assert noise.dtype == torch.float32


def test_perlin_noise_deterministic_with_seed() -> None:
    """Verify reproducibility with fixed seed."""
    torch.manual_seed(42)
    n1 = generate_perlin_noise(64, 64)
    torch.manual_seed(42)
    n2 = generate_perlin_noise(64, 64)
    assert torch.allclose(n1, n2)


def test_single_image_shape() -> None:
    """Verify single image output shape with default params."""
    gen = PerlinAnomalyGenerator(probability=1.0)
    img = torch.rand(3, 64, 64)
    aug, mask = gen(img)
    assert aug.shape == (3, 64, 64)
    assert mask.shape == (1, 64, 64)


def test_batch_shape() -> None:
    """Verify batch output shape with default params."""
    gen = PerlinAnomalyGenerator(probability=1.0)
    batch = torch.rand(4, 3, 64, 64)
    aug, mask = gen(batch)
    assert aug.shape == (4, 3, 64, 64)
    assert mask.shape == (4, 1, 64, 64)


def test_mask_binary() -> None:
    """Verify mask contains only 0 and 1 values."""
    gen = PerlinAnomalyGenerator(probability=1.0)
    img = torch.rand(3, 64, 64)
    _, mask = gen(img)
    unique_vals = torch.unique(mask)
    assert all(v in {0.0, 1.0} for v in unique_vals.tolist())


def test_probability_zero_returns_zeros_mask() -> None:
    """Verify probability=0 produces identity transform."""
    gen = PerlinAnomalyGenerator(probability=0.0)
    img = torch.rand(3, 64, 64)
    aug, mask = gen(img)
    assert torch.allclose(aug, img)
    assert mask.sum() == 0


def test_dual_mask_single_image_shape() -> None:
    """Verify dual_mask output shape for single image."""
    gen = PerlinAnomalyGenerator(probability=1.0, dual_mask=True)
    img = torch.rand(3, 64, 64)
    aug, mask = gen(img)
    assert aug.shape == (3, 64, 64)
    assert mask.shape == (1, 64, 64)


def test_dual_mask_batch_shape() -> None:
    """Verify dual_mask output shape for batch."""
    gen = PerlinAnomalyGenerator(probability=1.0, dual_mask=True)
    batch = torch.rand(4, 3, 64, 64)
    aug, mask = gen(batch)
    assert aug.shape == (4, 3, 64, 64)
    assert mask.shape == (4, 1, 64, 64)


def test_dual_mask_dtype_float() -> None:
    """Verify dual_mask produces float32 mask."""
    gen = PerlinAnomalyGenerator(probability=1.0, dual_mask=True)
    img = torch.rand(3, 64, 64)
    _, mask = gen(img)
    assert mask.dtype == torch.float32


def test_dual_mask_binary_after_union() -> None:
    """Mask must remain binary {0, 1} even after union/single logic."""
    gen = PerlinAnomalyGenerator(probability=1.0, dual_mask=True)
    img = torch.rand(3, 128, 128)
    for _ in range(20):
        _, mask = gen(img)
        unique_vals = torch.unique(mask)
        assert all(v in {0.0, 1.0} for v in unique_vals.tolist()), (
            f"Non-binary mask values found: {unique_vals.tolist()}"
        )


def test_dual_mask_can_produce_nonempty_mask() -> None:
    """Over many trials, dual_mask should produce at least one non-empty mask."""
    gen = PerlinAnomalyGenerator(probability=1.0, dual_mask=True)
    img = torch.rand(3, 64, 64)
    found_nonempty = False
    for _ in range(50):
        _, mask = gen(img)
        if mask.sum() > 0:
            found_nonempty = True
            break
    assert found_nonempty, "dual_mask=True never produced a non-empty mask in 50 trials"


def test_dual_mask_union_produces_larger_mask_on_average() -> None:
    """Union of two masks should (on average) cover more area than single mask."""
    torch.manual_seed(0)
    gen_single = PerlinAnomalyGenerator(probability=1.0, dual_mask=False)
    gen_dual = PerlinAnomalyGenerator(probability=1.0, dual_mask=True)
    img = torch.rand(3, 64, 64)

    single_areas = []
    dual_areas = []
    for _ in range(100):
        _, m = gen_single(img)
        single_areas.append(m.sum().item())
        _, m = gen_dual(img)
        dual_areas.append(m.sum().item())

    avg_single = sum(single_areas) / len(single_areas)
    avg_dual = sum(dual_areas) / len(dual_areas)
    assert avg_dual >= avg_single * 0.9, (
        f"Dual mask avg area ({avg_dual:.1f}) unexpectedly smaller than single ({avg_single:.1f})"
    )
