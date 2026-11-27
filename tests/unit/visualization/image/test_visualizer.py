# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``ImageVisualizer``."""

import numpy as np
import pytest
from PIL import Image

from anomalib.data import NumpyImageItem
from anomalib.visualization.image import ImageVisualizer


@pytest.mark.parametrize(
    ("kwargs", "expected_fields", "expected_panels"),
    [
        ({}, ["image", "gt_mask"], 2),  # default: image + 1 overlay (gt_mask absent)
        ({"fields": None}, None, 1),  # None disables individual fields; overlay only
        ({"fields": []}, [], 1),  # empty list disables individual fields; overlay only
        ({"fields": ["image", "anomaly_map"]}, ["image", "anomaly_map"], 3),
    ],
)
def test_image_visualizer_fields_handling(
    kwargs: dict[str, object],
    expected_fields: list[str] | None,
    expected_panels: int,
) -> None:
    """Preserves default fields while honoring explicit ``None`` and ``[]``."""
    item = NumpyImageItem(
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        anomaly_map=np.zeros((32, 32), dtype=np.float32),
    )
    visualizer = ImageVisualizer(field_size=(32, 32), overlay_fields=[("image", ["anomaly_map"])], **kwargs)
    assert visualizer.fields == expected_fields

    result = visualizer.visualize(item)
    assert isinstance(result, Image.Image)
    assert result.width == expected_panels * 32
