# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test the PostProcessor class."""

import pytest
import torch

from anomalib.data import ImageBatch
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
            (torch.tensor([20, 40, 60, 80]), 50, 50, 50, torch.tensor([0.0, 0.0, 1.0, 1.0])),  # all same
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


class TestEffectiveThreshold:
    """Tests for _effective_threshold and sensitivity-driven forward pass."""

    @staticmethod
    @pytest.mark.parametrize(
        ("threshold", "norm_min", "norm_max", "sensitivity", "expected"),
        [
            # sensitivity=0.5 → no shift
            (0.5, 0.0, 1.0, 0.5, 0.5),
            # sensitivity=1.0 → shift down by half range
            (0.5, 0.0, 1.0, 1.0, 0.0),
            # sensitivity=0.0 → shift up by half range
            (0.5, 0.0, 1.0, 0.0, 1.0),
            # non-unit range
            (0.3, 0.1, 0.9, 0.5, 0.3),
            # clamping: sensitivity > 1 treated as 1
            (0.5, 0.0, 1.0, 2.0, 0.0),
            # clamping: sensitivity < 0 treated as 0
            (0.5, 0.0, 1.0, -1.0, 1.0),
            # arbitrary values
            (0.4, 0.2, 0.8, 0.7, 0.4 + (0.8 - 0.2) * (0.5 - 0.7)),
        ],
    )
    def test_effective_threshold(
        threshold: float,
        norm_min: float,
        norm_max: float,
        sensitivity: float,
        expected: float,
    ) -> None:
        """Test _effective_threshold computes correct shifted threshold."""
        result = PostProcessor._effective_threshold(  # noqa: SLF001
            torch.tensor(threshold),
            torch.tensor(norm_min),
            torch.tensor(norm_max),
            torch.tensor(sensitivity),
        )
        assert result.item() == pytest.approx(expected, abs=1e-6)

    @staticmethod
    def test_forward_with_sensitivity_override() -> None:
        """Test that forward accepts sensitivity tensors without error."""
        processor = PostProcessor()
        processor._image_threshold.copy_(torch.tensor(0.5))  # noqa: SLF001
        processor._pixel_threshold.copy_(torch.tensor(0.5))  # noqa: SLF001
        processor.image_min.copy_(torch.tensor(0.0))
        processor.image_max.copy_(torch.tensor(1.0))
        processor.pixel_min.copy_(torch.tensor(0.0))
        processor.pixel_max.copy_(torch.tensor(1.0))

        batch = ImageBatch(
            image=torch.rand(2, 3, 4, 4),
            pred_score=torch.tensor([0.3, 0.7]),
            anomaly_map=torch.rand(2, 1, 4, 4),
        )
        result = processor(batch, image_sensitivity=torch.tensor(0.8), pixel_sensitivity=torch.tensor(0.8))
        assert result.pred_label is not None
        assert result.pred_mask is not None

    @staticmethod
    def test_forward_sensitivity_changes_labels() -> None:
        """High sensitivity should label more as anomalous (lower effective threshold)."""
        processor = PostProcessor()
        processor._image_threshold.copy_(torch.tensor(0.5))  # noqa: SLF001
        processor._pixel_threshold.copy_(torch.tensor(0.5))  # noqa: SLF001
        processor.image_min.copy_(torch.tensor(0.0))
        processor.image_max.copy_(torch.tensor(1.0))
        processor.pixel_min.copy_(torch.tensor(0.0))
        processor.pixel_max.copy_(torch.tensor(1.0))

        batch = ImageBatch(
            image=torch.rand(4, 3, 4, 4),
            pred_score=torch.tensor([0.3, 0.4, 0.6, 0.7]),
            anomaly_map=torch.rand(4, 1, 4, 4),
        )

        # Default sensitivity (0.5) → threshold stays at 0.5
        result_default = processor(batch, image_sensitivity=torch.tensor(0.5))
        # High sensitivity (0.9) → threshold shifts down
        result_high = processor(batch, image_sensitivity=torch.tensor(0.9))

        # More anomalous labels with high sensitivity
        assert result_high.pred_label.sum() >= result_default.pred_label.sum()

    @staticmethod
    def test_apply_threshold_none_input() -> None:
        """_apply_threshold returns None when pred_score is None."""
        result = PostProcessor._apply_threshold(None, torch.tensor(0.5))  # noqa: SLF001
        assert result is None
