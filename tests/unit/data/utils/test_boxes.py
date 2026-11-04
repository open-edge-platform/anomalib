# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for bounding box coordinate scaling."""

import pytest
import torch

from anomalib.data.utils.boxes import scale_boxes


@pytest.mark.parametrize(
    ("image_size", "new_size", "expected"),
    [
        ((100, 200), (200, 300), [[15.0, 40.0, 60.0, 120.0]]),
        ((100, 200), (50, 50), [[2.5, 10.0, 10.0, 30.0]]),
        ((100, 200), (200, 400), [[20.0, 40.0, 80.0, 120.0]]),
        ((100, 200), (100, 200), [[10.0, 20.0, 40.0, 60.0]]),
    ],
    ids=["unequal-upscale", "unequal-downscale", "uniform-scale", "unchanged"],
)
def test_scale_boxes_uses_width_for_x_and_height_for_y(
    image_size: tuple[int, int],
    new_size: tuple[int, int],
    expected: list[list[float]],
) -> None:
    """XYXY coordinates follow width and height, respectively."""
    boxes = torch.tensor([[10.0, 20.0, 40.0, 60.0]])
    original = boxes.clone()

    scaled = scale_boxes(boxes, torch.Size(image_size), torch.Size(new_size))

    torch.testing.assert_close(scaled, torch.tensor(expected))
    torch.testing.assert_close(boxes, original)
