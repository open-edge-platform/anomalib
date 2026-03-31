# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utilities to generate synthetic data.

This module provides utilities for generating synthetic data for anomaly detection.
The utilities include:

- Perlin noise generation: Functions for creating Perlin noise patterns
- Anomaly generation: Classes for generating synthetic anomalies

Example:
    >>> from anomalib.data.utils.generators import generate_perlin_noise
    >>> # Generate 256x256 Perlin noise
    >>> noise = generate_perlin_noise(256, 256)
    >>> print(noise.shape)
    torch.Size([256, 256])

    >>> from anomalib.data.utils.generators import PerlinAnomalyGenerator
    >>> # Create anomaly generator
    >>> generator = PerlinAnomalyGenerator()
    >>> # Generate anomaly mask
    >>> mask = generator.generate(256, 256)
"""

from .cutpaste import CutPasteGenerator
from .perlin import (
    DEFAULT_PERLIN_SCALE_EXPONENT_RANGE,
    GLASS_PERLIN_SCALE_EXPONENT_RANGE,
    PerlinAnomalyGenerator,
    apply_perlin_threshold_rescale,
    generate_perlin_noise,
    generate_perlin_noise_glass,
)

__all__ = [
    "CutPasteGenerator",
    "PerlinAnomalyGenerator",
    "generate_perlin_noise",
    "generate_perlin_noise_glass",
    "apply_perlin_threshold_rescale",
    "DEFAULT_PERLIN_SCALE_EXPONENT_RANGE",
    "GLASS_PERLIN_SCALE_EXPONENT_RANGE",
]
