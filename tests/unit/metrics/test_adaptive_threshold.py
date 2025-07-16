# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the adaptive threshold metric."""

import pytest
import torch

from anomalib.metrics.threshold.f1_adaptive_threshold import _F1AdaptiveThreshold


@pytest.mark.parametrize(
    ("labels", "preds", "target_threshold"),
    [
        (torch.tensor([0, 0, 0, 1, 1]), torch.tensor([2.3, 1.6, 2.6, 7.9, 3.3]), 3.3),  # standard case
        (torch.tensor([1, 0, 0, 0]), torch.tensor([4, 3, 2, 1]), 4),  # 100% recall for all thresholds
    ],
)
def test_adaptive_threshold(labels: torch.Tensor, preds: torch.Tensor, target_threshold: int | float) -> None:
    """Test if the adaptive threshold computation returns the desired value."""
    adaptive_threshold = _F1AdaptiveThreshold()
    adaptive_threshold.update(preds, labels)
    threshold_value = adaptive_threshold.compute()

    assert threshold_value == target_threshold
