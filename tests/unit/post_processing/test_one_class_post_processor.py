# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test the OneClassPostProcessor class."""

import pytest
import torch

from anomalib.data import InferenceBatch
from anomalib.post_processing.one_class import OneClassPostProcessor


class TestOneClassPostProcessor:
    """Test the forward pass of OneClassPostProcessor."""

    @staticmethod
    @pytest.mark.parametrize("enable_normalization", [True, False])
    def test_forward_keeps_pred_score_of_a_batch(enable_normalization: bool) -> None:
        """A batch of image scores must be used as given."""
        pred_score = torch.tensor([0.2, 0.6, 0.1, 0.9])
        anomaly_map = torch.rand(4, 1, 8, 8)
        post_processor = OneClassPostProcessor(enable_normalization=enable_normalization)

        output = post_processor(InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map))

        assert torch.equal(output.pred_score, pred_score)

    @staticmethod
    def test_forward_keeps_zero_pred_score() -> None:
        """A score of 0.0 is a valid score and must not be replaced by the anomaly map maximum."""
        pred_score = torch.tensor([0.0])
        anomaly_map = torch.full((1, 1, 8, 8), 5.0)
        post_processor = OneClassPostProcessor()

        output = post_processor(InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map))

        assert torch.equal(output.pred_score, pred_score)

    @staticmethod
    def test_forward_derives_pred_score_from_anomaly_map() -> None:
        """Without an image score, the maximum of each anomaly map is used."""
        anomaly_map = torch.rand(4, 1, 8, 8)
        post_processor = OneClassPostProcessor(enable_thresholding=False)

        output = post_processor(InferenceBatch(anomaly_map=anomaly_map))

        assert torch.equal(output.pred_score, torch.amax(anomaly_map, dim=(-2, -1)))

    @staticmethod
    def test_forward_requires_pred_score_or_anomaly_map() -> None:
        """Without a score and an anomaly map there is nothing to post-process."""
        with pytest.raises(ValueError, match="At least one of pred_score or anomaly_map"):
            OneClassPostProcessor()(InferenceBatch())
