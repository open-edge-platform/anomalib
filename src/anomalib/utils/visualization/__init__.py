# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tools for visualizing anomaly detection results.

This module provides utilities for visualizing anomaly detection outputs. The
utilities include:
    - Base visualization interface and common functionality
    - Image-based visualization for detection results
    - Explanation visualization for model interpretability
    - Metrics visualization for performance analysis

Example:
    >>> from anomalib.utils.visualization import ImageVisualizer
    >>> # Create visualizer for detection results
    >>> visualizer = ImageVisualizer()
    >>> # Visualize detection on an image
    >>> vis_result = visualizer.visualize(
    ...     image=image,
    ...     pred_mask=mask,
    ...     anomaly_map=heatmap
    ... )

The module ensures consistent and informative visualization across different
detection approaches and result types.
"""

from .base import GeneratorResult, VisualizationStep, Visualizer
from .explanation import ExplanationVisualizer
from .image import ImageResult, ImageVisualizer
from .metrics import MetricsVisualizer

__all__ = [
    "Visualizer",
    "ExplanationVisualizer",
    "ImageResult",
    "ImageVisualizer",
    "GeneratorResult",
    "MetricsVisualizer",
    "VisualizationStep",
]
