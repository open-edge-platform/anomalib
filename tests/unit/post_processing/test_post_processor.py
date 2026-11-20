# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test the PostProcessor class."""

import pytest
import torch

from anomalib.data import ImageBatch, InferenceBatch
from anomalib.post_processing import PostProcessor


class TestPostProcessor:
    """Test the PreProcessor class."""

    @staticmethod
    @pytest.mark.parametrize(
        ("preds", "min_val", "max_val", "thresh", "target"),
        [
            (torch.tensor([20, 40, 60, 80]), 0, 100, 50, torch.tensor([0.2, 0.4, 0.6, 0.8])),
            (torch.tensor([20, 40, 60, 80]), 0, 100, 40, torch.tensor([0.3, 0.5, 0.7, 0.9])),  # lower threshold
            (torch.tensor([20, 40, 60, 80]), 0, 100, 60, torch.tensor([0.1, 0.3, 0.5, 0.7])),  # higher threshold
            (torch.tensor([0, 40, 80, 120]), 20, 100, 50, torch.tensor([0.0, 0.375, 0.875, 1.0])),  # out of bounds
            (torch.tensor([-80, -60, -40, -20]), -100, 0, -50, torch.tensor([0.2, 0.4, 0.6, 0.8])),  # negative values
            (torch.tensor([20, 40, 60, 80]), 0, 100, -50, torch.tensor([1.0, 1.0, 1.0, 1.0])),  # threshold below range
            (torch.tensor([20, 40, 60, 80]), 0, 100, 150, torch.tensor([0.0, 0.0, 0.0, 0.0])),  # threshold above range
            (torch.tensor([20, 40, 50, 60, 80]), 50, 50, 50, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])),  # all same
            (torch.tensor(60), 0, 100, 50, torch.tensor(0.6)),  # scalar tensor
            (torch.tensor([[20, 40], [60, 80]]), 0, 100, 50, torch.tensor([[0.2, 0.4], [0.6, 0.8]])),  # 2D tensor
        ],
    )
    def test_normalize(
        preds: torch.Tensor,
        min_val: float,
        max_val: float,
        thresh: float,
        target: torch.Tensor,
    ) -> None:
        """Test the normalize method."""
        pre_processor = PostProcessor()
        normalized = pre_processor._normalize(  # noqa: SLF001
            preds,
            torch.tensor(min_val),
            torch.tensor(max_val),
            torch.tensor(thresh),
        )
        assert torch.allclose(normalized, target)

    @staticmethod
    def test_normalize_returns_predictions_when_stats_are_unset() -> None:
        """Uninitialized normalization stats leave predictions unchanged."""
        preds = torch.tensor([0.2, 0.8])
        normalized = PostProcessor._normalize(  # noqa: SLF001
            preds,
            torch.tensor(float("nan")),
            torch.tensor(float("nan")),
            torch.tensor(float("nan")),
        )
        assert torch.equal(normalized, preds)

    @staticmethod
    def test_normalize_uses_midpoint_when_threshold_is_unset() -> None:
        """An unset threshold defaults to midpoint of normalization range."""
        normalized = PostProcessor._normalize(  # noqa: SLF001
            torch.tensor([0.0, 1.0]),
            torch.tensor(0.0),
            torch.tensor(1.0),
            torch.tensor(float("nan")),
        )
        assert torch.equal(normalized, torch.tensor([0.0, 1.0]))

    @staticmethod
    @pytest.mark.parametrize(
        ("preds", "thresh", "target"),
        [
            (torch.tensor(20), 50, torch.tensor(0).bool()),  # test scalar
            (torch.tensor([20, 40, 60, 80]), 50, torch.tensor([0, 0, 1, 1]).bool()),  # test 1d tensor
            (torch.tensor([[20, 40], [60, 80]]), 50, torch.tensor([[0, 0], [1, 1]]).bool()),  # test 2d tensor
            (torch.tensor(50), 50, torch.tensor(0).bool()),  # test on threshold labeled as normal
            (torch.tensor([-80, -60, -40, -20]), -50, torch.tensor([0, 0, 1, 1]).bool()),  # test negative
        ],
    )
    def test_apply_threshold(preds: torch.Tensor, thresh: float, target: torch.Tensor) -> None:
        """Test the apply_threshold method."""
        pre_processor = PostProcessor()
        binary_preds = pre_processor._apply_threshold(preds, torch.tensor(thresh))  # noqa: SLF001
        assert torch.allclose(binary_preds, target)

    @staticmethod
    def test_apply_threshold_returns_predictions_when_threshold_is_unset() -> None:
        """An unset threshold leaves predictions unchanged outside export tracing."""
        preds = torch.tensor([0.2, 0.8])
        result = PostProcessor._apply_threshold(preds, torch.tensor(float("nan")))  # noqa: SLF001
        assert torch.equal(result, preds)

    @staticmethod
    def _fit_without_normalization() -> PostProcessor:
        """Return a post-processor that learned thresholds of 8 on raw scores."""
        scores = torch.tensor([3.0, 4.0, 8.0, 9.0])
        batch = ImageBatch(
            image=torch.rand(4, 3, 4, 4),
            pred_score=scores,
            gt_label=torch.tensor([0, 0, 1, 1]),
            anomaly_map=scores.reshape(4, 1, 1) * torch.ones(4, 4, 4),
            gt_mask=torch.tensor([0, 0, 1, 1]).reshape(4, 1, 1) * torch.ones(4, 4, 4, dtype=torch.long),
        )
        post_processor = PostProcessor(enable_normalization=False)
        post_processor.on_validation_batch_end(None, None, batch)
        post_processor.on_validation_epoch_end(None, None)
        return post_processor

    def test_learned_threshold_applied_to_batch_without_normalization(self) -> None:
        """Raw scores are compared with the learned threshold, not with the normalized one."""
        post_processor = self._fit_without_normalization()
        scores = torch.tensor([3.0, 4.0, 8.5, 9.0])
        batch = ImageBatch(
            image=torch.rand(4, 3, 4, 4),
            pred_score=scores,
            anomaly_map=scores.reshape(4, 1, 1) * torch.ones(4, 4, 4),
        )
        post_processor.post_process_batch(batch)
        assert batch.pred_label.tolist() == [False, False, True, True]
        assert batch.pred_mask.reshape(4, -1).all(dim=1).tolist() == [False, False, True, True]

    def test_learned_threshold_applied_in_forward_without_normalization(self) -> None:
        """Forward labels raw scores with the learned threshold."""
        scores = torch.tensor([3.0, 4.0, 8.5, 9.0])
        anomaly_map = scores.reshape(4, 1, 1) * torch.ones(4, 4, 4)
        predictions = self._fit_without_normalization()(InferenceBatch(pred_score=scores, anomaly_map=anomaly_map))
        assert predictions.pred_label.tolist() == [False, False, True, True]
        assert predictions.pred_mask.reshape(4, -1).all(dim=1).tolist() == [False, False, True, True]

    def test_forward_derives_score_from_map_without_normalization(self) -> None:
        """The score derived from the anomaly map is kept when normalization is disabled."""
        anomaly_map = torch.tensor([3.0, 9.0]).reshape(2, 1, 1) * torch.ones(2, 4, 4)
        predictions = self._fit_without_normalization()(InferenceBatch(anomaly_map=anomaly_map))
        assert predictions.pred_score.tolist() == [3.0, 9.0]
        assert predictions.pred_label.tolist() == [False, True]

    @staticmethod
    def test_unlearned_threshold_keeps_normalized_scale_without_normalization() -> None:
        """Without a learned threshold the sensitivity-based threshold is still used."""
        post_processor = PostProcessor(enable_normalization=False, image_sensitivity=0.25)
        assert post_processor.image_decision_threshold == pytest.approx(0.75)

    @staticmethod
    def test_decision_threshold_is_normalized_with_normalization() -> None:
        """With normalization enabled the decision threshold stays on the normalized scale."""
        post_processor = PostProcessor(image_sensitivity=0.25, pixel_sensitivity=0.4)
        post_processor._image_threshold.fill_(8.0)  # noqa: SLF001
        post_processor._pixel_threshold.fill_(8.0)  # noqa: SLF001
        assert post_processor.image_decision_threshold == pytest.approx(0.75)
        assert post_processor.pixel_decision_threshold == pytest.approx(0.6)

    @staticmethod
    def test_thresholds_computed() -> None:
        """Test that both image and pixel threshold are computed correctly."""
        batch = ImageBatch(
            image=torch.rand(4, 3, 3, 3),
            anomaly_map=torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]]),
            gt_mask=torch.tensor([[0, 0, 0], [0, 0, 0], [0, 1, 1]]),
            pred_score=torch.tensor([20, 40, 60, 80]),
            gt_label=torch.tensor([0, 0, 1, 1]),
        )
        pre_processor = PostProcessor()
        pre_processor.on_validation_batch_end(None, None, batch)
        pre_processor.on_validation_epoch_end(None, None)
        assert pre_processor.image_threshold == 60
        assert pre_processor.pixel_threshold == 80

    @staticmethod
    def test_pixel_threshold_matching() -> None:
        """Test that pixel_threshold is used as image threshold when no gt masks are available."""
        batch = ImageBatch(
            image=torch.rand(4, 3, 10, 10),
            anomaly_map=torch.rand(4, 10, 10),
            pred_score=torch.tensor([20, 40, 60, 80]),
            gt_label=torch.tensor([0, 0, 1, 1]),
        )
        pre_processor = PostProcessor(enable_threshold_matching=True)
        pre_processor.on_validation_batch_end(None, None, batch)
        pre_processor.on_validation_epoch_end(None, None)
        assert pre_processor.image_threshold == pre_processor.pixel_threshold

    @staticmethod
    def test_image_threshold_matching() -> None:
        """Test that pixel_threshold is used as image threshold when no gt masks are available."""
        batch = ImageBatch(
            image=torch.rand(4, 3, 3, 3),
            anomaly_map=torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]]),
            gt_mask=torch.tensor([[0, 0, 0], [0, 0, 0], [0, 1, 1]]),
            pred_score=torch.tensor([20, 40, 60, 80]),
        )
        pre_processor = PostProcessor(enable_threshold_matching=True)
        pre_processor.on_validation_batch_end(None, None, batch)
        pre_processor.on_validation_epoch_end(None, None)
        assert pre_processor.image_threshold == pre_processor.pixel_threshold
