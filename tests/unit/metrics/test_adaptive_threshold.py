# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the adaptive threshold metric."""

import logging

import pytest
import torch
from torchmetrics.classification import BinaryPrecisionRecallCurve

from anomalib.metrics.threshold.f1_adaptive_threshold import _F1AdaptiveThreshold


class TestF1AdaptiveThresholdNonBinned:
    """Test F1AdaptiveThreshold with default settings (non-binned mode)."""

    @staticmethod
    @pytest.mark.parametrize(
        ("labels", "preds", "target_threshold"),
        [
            (torch.tensor([0, 0, 0, 1, 1]), torch.tensor([2.3, 1.6, 2.6, 7.9, 3.3]), 3.3),  # standard case
            (torch.tensor([1, 0, 0, 0]), torch.tensor([4, 3, 2, 1]), 4),  # 100% recall for all thresholds
            (
                torch.tensor([1, 1, 1, 1]),
                torch.tensor([4, 3, 2, 1]),
                1,
            ),  # use minimum value when all images are anomalous
            (torch.tensor([0, 0, 0, 0]), torch.tensor([4, 3, 2, 1]), 4),  # use maximum value when all images are normal
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

        assert threshold_value == pytest.approx(target_threshold)

    @staticmethod
    def test_no_anomalous_samples_warning(caplog: pytest.LogCaptureFixture) -> None:
        """Test warning is logged when no anomalous samples exist (non-binned)."""
        labels = torch.tensor([0, 0, 0, 0])
        preds = torch.tensor([0.1, 0.2, 0.3, 0.4])

        adaptive_threshold = _F1AdaptiveThreshold()
        adaptive_threshold.update(preds, labels)

        with caplog.at_level(logging.WARNING):
            threshold_value = adaptive_threshold.compute()

        assert "validation set does not contain any anomalous images" in caplog.text
        assert threshold_value == pytest.approx(preds.max().item())

    @staticmethod
    def test_no_normal_samples_warning(caplog: pytest.LogCaptureFixture) -> None:
        """Test warning is logged when no normal samples exist (non-binned)."""
        labels = torch.tensor([1, 1, 1, 1])
        preds = torch.tensor([0.5, 0.6, 0.7, 0.8])

        adaptive_threshold = _F1AdaptiveThreshold()
        adaptive_threshold.update(preds, labels)

        with caplog.at_level(logging.WARNING):
            threshold_value = adaptive_threshold.compute()

        assert "validation set does not contain any normal images" in caplog.text
        assert threshold_value == pytest.approx(preds.min().item())


class TestF1AdaptiveThresholdBinned:
    """Test F1AdaptiveThreshold with pre-specified thresholds (binned mode)."""

    @staticmethod
    @pytest.mark.parametrize(
        "thresholds",
        [
            10,
            [0.0, 0.25, 0.5, 0.75, 1.0],
            torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]),
        ],
    )
    def test_compute_returns_expected_threshold(
        thresholds: int | list[float] | torch.Tensor,
    ) -> None:
        """Test F1AdaptiveThreshold returns a threshold that maximizes F1."""
        labels = torch.tensor([0, 0, 0, 1, 1])
        preds = torch.tensor([0.1, 0.2, 0.3, 0.8, 0.9])

        adaptive_threshold = _F1AdaptiveThreshold(thresholds=thresholds)
        adaptive_threshold.update(preds, labels)
        threshold_value = adaptive_threshold.compute()

        pr_curve = BinaryPrecisionRecallCurve(thresholds=thresholds)
        pr_curve.update(preds, labels)
        precision, recall, candidate_thresholds = pr_curve.compute()
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
        f1_scores = torch.nan_to_num(f1_scores, nan=0.0)
        f1_at_thresholds = f1_scores[: len(candidate_thresholds)]
        max_f1 = f1_at_thresholds.max()

        threshold_idx = torch.argmin(torch.abs(candidate_thresholds - threshold_value))
        assert f1_scores[threshold_idx].item() == pytest.approx(max_f1.item())

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
    def test_no_anomalous_samples_returns_max_threshold(caplog: pytest.LogCaptureFixture) -> None:
        """Test no-anomalous returns highest candidate threshold in binned mode."""
        labels = torch.tensor([0, 0, 0, 0, 0])
        preds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        thresholds = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

        adaptive_threshold = _F1AdaptiveThreshold(thresholds=thresholds)
        adaptive_threshold.update(preds, labels)

        with caplog.at_level(logging.WARNING):
            threshold_value = adaptive_threshold.compute()

        assert threshold_value == pytest.approx(thresholds[-1].item())

    @staticmethod
    def test_no_normal_samples_warning(caplog: pytest.LogCaptureFixture) -> None:
        """Test warning is logged when no normal samples exist (binned mode)."""
        labels = torch.tensor([1, 1, 1, 1, 1])
        preds = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
        thresholds = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

        adaptive_threshold = _F1AdaptiveThreshold(thresholds=thresholds)
        adaptive_threshold.update(preds, labels)

        with caplog.at_level(logging.WARNING):
            threshold_value = adaptive_threshold.compute()

        assert "validation set does not contain any normal images" in caplog.text
        assert threshold_value == pytest.approx(thresholds[0].item())

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
        assert "validation set does not contain any normal images" not in caplog.text
