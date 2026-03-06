# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the adaptive threshold metric."""

import logging

import pytest
import torch

from anomalib.metrics.threshold.f1_adaptive_threshold import _F1AdaptiveThreshold


class TestF1AdaptiveThresholdNonBinned:
    """Test F1AdaptiveThreshold with default settings (non-binned mode)."""

    @staticmethod
    @pytest.mark.parametrize(
        ("labels", "preds", "target_threshold"),
        [
            (torch.tensor([0, 0, 0, 1, 1]), torch.tensor([2.3, 1.6, 2.6, 7.9, 3.3]), 3.3),  # standard case
            (torch.tensor([1, 0, 0, 0]), torch.tensor([4, 3, 2, 1]), 4),  # 100% recall for all thresholds
        ],
    )
    def test_adaptive_threshold(
        labels: torch.Tensor,
        preds: torch.Tensor,
        target_threshold: int | float,
    ) -> None:
        """Test if the adaptive threshold computation returns the desired value."""
        adaptive_threshold = _F1AdaptiveThreshold()
        adaptive_threshold.update(preds, labels)
        threshold_value = adaptive_threshold.compute()

        assert threshold_value == target_threshold


class TestF1AdaptiveThresholdBinned:
    """Test F1AdaptiveThreshold with pre-specified thresholds (binned mode)."""

    @staticmethod
    @pytest.mark.parametrize(
        ("thresholds", "expected"),
        [
            (10, 1.0),
            ([0.0, 0.25, 0.5, 0.75, 1.0], 1.0),
            (torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]), 1.0),
        ],
    )
    def test_compute_returns_expected_threshold(
        thresholds: int | list[float] | torch.Tensor,
        expected: float,
    ) -> None:
        """Test F1AdaptiveThreshold returns correct threshold for different threshold types."""
        labels = torch.tensor([0, 0, 0, 1, 1])
        preds = torch.tensor([0.1, 0.2, 0.3, 0.8, 0.9])

        adaptive_threshold = _F1AdaptiveThreshold(thresholds=thresholds)
        adaptive_threshold.update(preds, labels)
        threshold_value = adaptive_threshold.compute()

        assert threshold_value == expected

    @staticmethod
    def test_no_anomalous_samples_warning(caplog: pytest.LogCaptureFixture) -> None:
        """Test warning is logged when no anomalous samples exist."""
        labels = torch.tensor([0, 0, 0, 0, 0])
        preds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

        adaptive_threshold = _F1AdaptiveThreshold(thresholds=10)
        adaptive_threshold.update(preds, labels)

        with caplog.at_level(logging.WARNING):
            _ = adaptive_threshold.compute()

        assert "validation set does not contain any anomalous images" in caplog.text

    @staticmethod
    def test_anomalous_samples_no_warning(caplog: pytest.LogCaptureFixture) -> None:
        """Test no warning when anomalous samples exist."""
        labels = torch.tensor([0, 0, 0, 1, 1])
        preds = torch.tensor([0.1, 0.2, 0.3, 0.8, 0.9])

        adaptive_threshold = _F1AdaptiveThreshold(thresholds=10)
        adaptive_threshold.update(preds, labels)

        with caplog.at_level(logging.WARNING):
            _ = adaptive_threshold.compute()

        assert "validation set does not contain any anomalous images" not in caplog.text
