# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility Functions for the Tiled Ensemble Pipeline.

This module provides auxiliary functions and classes that support the core
components of the Tiled Ensemble approach.

Included Utilities:
    - `ensemble_engine`: Modified Anomalib Engine, adjusted to support ensemble training.
    - `ensemble_tiling`: Functions to divide images into tiles and manage their positions.
    - `helper_functions`: Miscellaneous helper functions used across the pipeline.
    - `prediction_data`: Data structure to store and manage predictions.
    - `prediction_merging`: Functions to merge predictions from different tiles.

Reference:
    Blaž Rolih, Dick Ameln, Ashwin Vaidya, Samet Akçay:
    "Divide and Conquer: High-Resolution Industrial Anomaly Detection via
    Memory Efficient Tiled Ensemble." Proceedings of the IEEE/CVF Conference
    on Computer Vision and Pattern Recognition Workshops (VAND 2.0). 2024.
"""

from enum import Enum


class NormalizationStage(str, Enum):
    """Enum signaling at which stage the normalization is done.

    In case of tile, tiles are normalized for each tile position separately.
    In case of image, normalization is done at the end when images are joined back together.
    In case of none, output is not normalized.
    """

    TILE = "tile"
    IMAGE = "image"
    NONE = "none"


class ThresholdingStage(str, Enum):
    """Enum signaling at which stage the thresholding is applied.

    In case of tile, thresholding is applied for each tile location separately.
    In case of image, thresholding is applied at the end when images are joined back together.
    """

    TILE = "tile"
    IMAGE = "image"


class PredictData(Enum):
    """Enum indicating which data to use in prediction job."""

    VAL = "val"
    TEST = "test"


__all__ = [
    "NormalizationStage",
    "ThresholdingStage",
    "PredictData",
]
