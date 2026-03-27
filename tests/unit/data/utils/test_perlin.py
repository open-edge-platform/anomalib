# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared Perlin generator utilities."""

import torch

from anomalib.data.utils.generators import (
    DEFAULT_PERLIN_SCALE_EXPONENT_RANGE,
    GLASS_PERLIN_SCALE_EXPONENT_RANGE,
    PerlinAnomalyGenerator,
    apply_perlin_threshold_rescale,
    generate_perlin_noise,
    generate_perlin_noise_glass,
)


def test_generate_perlin_noise_glass_shape() -> None:
    """Ensure GLASS-compatible helper returns expected shape."""
    noise = generate_perlin_noise_glass(height=63, width=95, device=torch.device("cpu"))
    assert noise.shape == (63, 95)


def test_apply_perlin_threshold_rescale_handles_uniform_noise() -> None:
    """Ensure helper safely handles uniform noise below threshold."""
    perlin_noise = torch.full((8, 8), -0.2)
    rescaled = apply_perlin_threshold_rescale(perlin_noise, threshold=0.5, enabled=True)
    assert torch.equal(rescaled, perlin_noise)


def test_perlin_anomaly_generator_accepts_compatibility_knobs() -> None:
    """Ensure PerlinAnomalyGenerator exposes GLASS compatibility params."""
    generator = PerlinAnomalyGenerator(
        perlin_pad_to_power_of_2=False,
        perlin_scale_exponent_range=GLASS_PERLIN_SCALE_EXPONENT_RANGE,
        perlin_rescale_below_threshold=False,
    )
    assert generator.perlin_pad_to_power_of_2 is False
    assert generator.perlin_scale_exponent_range == GLASS_PERLIN_SCALE_EXPONENT_RANGE
    assert generator.perlin_rescale_below_threshold is False


def test_default_scale_range_kept_for_standard_generator() -> None:
    """Ensure default exponent range remains unchanged."""
    noise = generate_perlin_noise(height=32, width=32, scale_exponent_range=DEFAULT_PERLIN_SCALE_EXPONENT_RANGE)
    assert noise.shape == (32, 32)
