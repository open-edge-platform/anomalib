# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Reusable DINOv2 transformer layers.

This package provides the low-level Vision Transformer building blocks
(``Attention``, ``MemEffAttention``, ``DropPath``, ``LayerScale``, ``Block``, ...)
used by the anomaly-detection models. DINOv2 backbones themselves are loaded via
``timm`` (see :class:`anomalib.models.components.feature_extractors.TimmFeatureExtractor`).

References:
    https://github.com/facebookresearch/dinov2/blob/main/dinov2/
"""

import warnings

warnings.warn(
    "The anomalib.models.components.dinov2 package is deprecated and will be removed in a future release. "
    "Please use the timm-based feature extractor instead: "
    "anomalib.models.components.feature_extractors.TimmFeatureExtractor",
    FutureWarning,
    stacklevel=2,
)
