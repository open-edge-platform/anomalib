# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SuperADD model.

Verifies that :class:`PatchedExecution` reassembles per-patch backbone outputs
into an exact full-resolution token map. A mock backbone encodes the global
pixel coordinates of every token, so any error in the patch layout, the
overlap splitting, or the batch/patch reshaping shows up as a nonzero
difference from the expected coordinate ramp.
"""

import itertools
from dataclasses import dataclass

import pytest
import torch
from torch import nn

from anomalib.models.image.super_add.post_processor import SuperADDPostProcessor
from anomalib.models.image.super_add.torch_model import PatchedExecution

MODEL_PATCH_SIZE = 16


class CoordinateBackbone(nn.Module):
    """Mock backbone returning the mean input value of each token cell.

    When fed an image whose channels contain global coordinates, every output
    token holds the exact center coordinate of its 16x16 pixel cell, allowing
    bit-exact verification of the stitched result.
    """

    @staticmethod
    def forward(x: torch.Tensor) -> list[torch.Tensor]:
        """Average-pool each token cell and return NLC tokens."""
        pooled = torch.nn.functional.avg_pool2d(x, MODEL_PATCH_SIZE)
        return [pooled.flatten(2).transpose(1, 2)]


def _coordinate_image(batch_size: int, height: int, width: int) -> torch.Tensor:
    ys = torch.arange(height).float().view(1, 1, height, 1).expand(batch_size, 1, height, width)
    xs = torch.arange(width).float().view(1, 1, 1, width).expand(batch_size, 1, height, width)
    ids = torch.arange(batch_size).float().view(batch_size, 1, 1, 1).expand(batch_size, 1, height, width) * 10000
    return torch.cat([ids, ys, xs], dim=1)


def _expected_tokens(batch_size: int, height: int, width: int) -> torch.Tensor:
    tokens_y, tokens_x = height // MODEL_PATCH_SIZE, width // MODEL_PATCH_SIZE
    center = (MODEL_PATCH_SIZE - 1) / 2
    ey = (torch.arange(tokens_y).float() * MODEL_PATCH_SIZE + center).view(tokens_y, 1).expand(tokens_y, tokens_x)
    ex = (torch.arange(tokens_x).float() * MODEL_PATCH_SIZE + center).view(1, tokens_x).expand(tokens_y, tokens_x)
    expected = []
    for batch_idx in range(batch_size):
        ids = torch.full((tokens_y, tokens_x), batch_idx * 10000.0)
        expected.append(torch.stack([ids, ey, ex], dim=-1))
    return torch.stack(expected)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize(
    ("height", "width", "patch_size", "patch_overlap"),
    [
        (1024, 1024, 512, 128),  # 3x3 grid, evenly spaced patches
        (896, 896, 448, 64),  # 3x3 grid, uneven strides
        (1024, 768, 448, 64),  # non-square image
        (1024, 1224, 512, 128),  # non-square, width not divisible by 16
    ],
)
def test_patched_execution_stitching_is_exact(
    batch_size: int,
    height: int,
    width: int,
    patch_size: int,
    patch_overlap: int,
) -> None:
    """Stitched token maps must exactly match a single-pass coordinate ramp."""
    patch_exec = PatchedExecution(
        CoordinateBackbone(),
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        model_patch_size=MODEL_PATCH_SIZE,
    )
    image = _coordinate_image(batch_size, height, width)

    result = patch_exec(image)[0]

    expected = _expected_tokens(batch_size, height, width)
    assert result.shape == expected.shape
    torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dim_size", [448, 535, 1024, 1224])
def test_axis_patch_split_covers_axis_without_gaps(dim_size: int) -> None:
    """Result ROIs must tile the token axis contiguously and completely."""
    patch_exec = PatchedExecution(
        CoordinateBackbone(),
        patch_size=448,
        patch_overlap=64,
        model_patch_size=MODEL_PATCH_SIZE,
    )
    input_rois, prediction_rois, result_rois = patch_exec.axis_patch_split(dim_size)

    assert result_rois[0][0] == 0
    assert result_rois[-1][1] == dim_size // MODEL_PATCH_SIZE
    for previous, current in itertools.pairwise(result_rois):
        assert previous[1] == current[0]
    for (pred_start, pred_end), (res_start, res_end) in zip(prediction_rois, result_rois, strict=True):
        assert pred_end - pred_start == res_end - res_start
    for in_start, in_end in input_rois:
        assert in_end - in_start == patch_exec.patch_size


@dataclass
class DummyValidationBatch:
    """Minimal batch carrying the fields the post-processor consumes."""

    anomaly_map: torch.Tensor
    pred_score: torch.Tensor


def test_percentile_post_processor_thresholds() -> None:
    """Thresholds must equal the configured percentile of validation scores times the factor."""
    post_processor = SuperADDPostProcessor(
        pixel_threshold_percentile=95.0,
        pixel_threshold_factor=1.421,
        image_threshold_percentile=95.0,
        image_threshold_factor=1.0,
    )

    torch.manual_seed(0)
    batches = [DummyValidationBatch(anomaly_map=torch.rand(2, 32, 32), pred_score=torch.rand(2)) for _ in range(3)]
    for batch in batches:
        post_processor.on_validation_batch_end(None, None, batch)
    post_processor.on_validation_epoch_end(None, None)

    all_pixels = torch.cat([batch.anomaly_map.flatten() for batch in batches])
    all_scores = torch.cat([batch.pred_score for batch in batches])
    expected_pixel = torch.quantile(all_pixels, 0.95) * 1.421
    expected_image = torch.quantile(all_scores, 0.95)

    torch.testing.assert_close(post_processor._pixel_threshold, expected_pixel)  # noqa: SLF001
    torch.testing.assert_close(post_processor._image_threshold, expected_image)  # noqa: SLF001
    # normalization statistics must still be computed by the base class
    assert post_processor.pixel_max.item() == all_pixels.max().item()
    assert post_processor.image_min.item() == all_scores.min().item()
