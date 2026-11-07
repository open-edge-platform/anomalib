# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MEBin algorithm and MEBinPostProcessor."""

import pytest
import torch

from anomalib.data import ImageBatch, InferenceBatch
from anomalib.post_processing import MEBinPostProcessor
from anomalib.post_processing.mebin import (
    _count_connected_components,
    _erode,
    _find_stable_threshold,
    mebin_binarize,
)


class TestCountConnectedComponents:
    """Tests for _count_connected_components."""

    @staticmethod
    def test_single_component() -> None:
        """A single connected blob should yield count = 1."""
        mask = torch.zeros(1, 1, 8, 8)
        mask[0, 0, 2:5, 2:5] = 1.0
        counts = _count_connected_components(mask)
        assert counts.item() == 1

    @staticmethod
    def test_two_components() -> None:
        """Two separated blobs should yield count = 2."""
        mask = torch.zeros(1, 1, 16, 16)
        mask[0, 0, 1:3, 1:3] = 1.0
        mask[0, 0, 10:12, 10:12] = 1.0
        counts = _count_connected_components(mask)
        assert counts.item() == 2

    @staticmethod
    def test_empty_mask() -> None:
        """All-zero mask should yield count = 0."""
        mask = torch.zeros(1, 1, 8, 8)
        counts = _count_connected_components(mask)
        assert counts.item() == 0

    @staticmethod
    def test_batch() -> None:
        """Counts should be computed per image in a batch."""
        mask = torch.zeros(2, 1, 16, 16)
        # Image 0: 1 component
        mask[0, 0, 2:5, 2:5] = 1.0
        # Image 1: 2 components
        mask[1, 0, 1:3, 1:3] = 1.0
        mask[1, 0, 10:12, 10:12] = 1.0
        counts = _count_connected_components(mask)
        assert counts[0].item() == 1
        assert counts[1].item() == 2


class TestErode:
    """Tests for _erode."""

    @staticmethod
    def test_small_region_removed() -> None:
        """A region smaller than the kernel should be eroded away."""
        mask = torch.zeros(1, 1, 16, 16)
        mask[0, 0, 7, 7] = 1.0  # single pixel
        eroded = _erode(mask, kernel_size=3)
        assert eroded.sum().item() == 0

    @staticmethod
    def test_large_region_survives() -> None:
        """A region larger than the kernel should partially survive."""
        mask = torch.zeros(1, 1, 32, 32)
        mask[0, 0, 8:24, 8:24] = 1.0  # 16x16 block
        eroded = _erode(mask, kernel_size=3)
        assert eroded.sum().item() > 0


class TestFindStableThreshold:
    """Tests for _find_stable_threshold."""

    @staticmethod
    def test_clear_stable_interval() -> None:
        """A clear plateau should be found."""
        # Simulate: threshold sweep produces [0,0,1,1,1,1,1,1,2,3,5]
        # The plateau of 1s has length 6.
        counts = [0, 0, 1, 1, 1, 1, 1, 1, 2, 3, 5]
        thresh, est_num = _find_stable_threshold(counts, min_interval_len=4, sample_rate=4)
        assert est_num == 1
        # Endpoint index = 7, threshold = 255 - 7*4 = 227
        assert thresh == 227

    @staticmethod
    def test_no_stable_interval() -> None:
        """When no plateau exists, should return (255, 0)."""
        counts = [0, 1, 2, 3, 4, 5]
        thresh, est_num = _find_stable_threshold(counts, min_interval_len=4, sample_rate=4)
        assert thresh == 255
        assert est_num == 0

    @staticmethod
    def test_all_zeros() -> None:
        """All-zero counts (no components) should return (255, 0)."""
        counts = [0, 0, 0, 0, 0]
        thresh, est_num = _find_stable_threshold(counts, min_interval_len=2, sample_rate=4)
        assert thresh == 255
        assert est_num == 0

    @staticmethod
    def test_multiple_plateaus_picks_longest() -> None:
        """When multiple stable intervals exist, the longest should win."""
        # Plateau of 2s (len=3), plateau of 1s (len=5)
        counts = [0, 2, 2, 2, 0, 1, 1, 1, 1, 1, 3]
        _, est_num = _find_stable_threshold(counts, min_interval_len=2, sample_rate=4)
        assert est_num == 1  # The plateau of 1s is longer


class TestMebinBinarize:
    """Tests for _mebin_binarize."""

    @staticmethod
    def test_output_shapes() -> None:
        """Output masks and thresholds should have correct shapes."""
        maps = torch.rand(4, 1, 32, 32)
        masks, thresholds = mebin_binarize(maps, sample_rate=8, min_interval_len=2)
        assert masks.shape == maps.shape
        assert thresholds.shape == (4,)

    @staticmethod
    def test_binary_values() -> None:
        """Masks should contain only 0 and 1."""
        maps = torch.rand(2, 1, 32, 32)
        masks, _ = mebin_binarize(maps, sample_rate=8, min_interval_len=2)
        unique = masks.unique()
        assert torch.all((unique == 0) | (unique == 1))

    @staticmethod
    def test_uniform_maps_no_anomaly() -> None:
        """Uniform anomaly maps should produce all-zero masks."""
        maps = torch.full((2, 1, 16, 16), 0.5)
        masks, _ = mebin_binarize(maps)
        assert masks.sum().item() == 0

    @staticmethod
    def test_clear_anomaly_detected() -> None:
        """A clear high-score region should be detected."""
        maps = torch.zeros(1, 1, 32, 32)
        # Create a clear anomaly region
        maps[0, 0, 10:20, 10:20] = 1.0
        masks, _ = mebin_binarize(maps, sample_rate=4, min_interval_len=2, erode=False)
        # The anomaly region should be at least partially detected
        assert masks[0, 0, 10:20, 10:20].sum().item() > 0
        # Background should remain mostly zero
        assert masks[0, 0, :5, :5].sum().item() == 0

    @staticmethod
    def test_invalid_shape() -> None:
        """Should raise ValueError for wrong input shapes."""
        with pytest.raises(ValueError, match="Expected anomaly_maps of shape"):
            mebin_binarize(torch.rand(4, 32, 32))

    @staticmethod
    def test_erode_flag() -> None:
        """Erosion should reduce the detected region."""
        maps = torch.zeros(1, 1, 64, 64)
        maps[0, 0, 20:44, 20:44] = 1.0
        masks_no_erode, _ = mebin_binarize(maps, erode=False, sample_rate=8, min_interval_len=2)
        masks_erode, _ = mebin_binarize(maps, erode=True, kernel_size=3, sample_rate=8, min_interval_len=2)
        # Erosion should either reduce or keep the same area
        assert masks_erode.sum() <= masks_no_erode.sum()

    @staticmethod
    def test_single_image() -> None:
        """Should work correctly with batch size 1."""
        maps = torch.rand(1, 1, 16, 16)
        masks, thresholds = mebin_binarize(maps, sample_rate=8, min_interval_len=2)
        assert masks.shape == (1, 1, 16, 16)
        assert thresholds.shape == (1,)


class TestMEBinPostProcessor:
    """Tests for MEBinPostProcessor."""

    @staticmethod
    def test_initialization_default_params() -> None:
        """Test default constructor parameters."""
        processor = MEBinPostProcessor()
        assert processor.sample_rate == 4
        assert processor.min_interval_len == 4
        assert processor.erode is True
        assert processor.kernel_size == 6

    @staticmethod
    @pytest.mark.parametrize(
        ("sample_rate", "min_interval_len", "erode", "kernel_size"),
        [
            (2, 3, True, 4),
            (8, 6, False, 8),
            (1, 1, True, 2),
        ],
    )
    def test_initialization_custom_params(
        sample_rate: int,
        min_interval_len: int,
        erode: bool,
        kernel_size: int,
    ) -> None:
        """Test custom constructor parameters are stored correctly."""
        processor = MEBinPostProcessor(
            sample_rate=sample_rate,
            min_interval_len=min_interval_len,
            erode=erode,
            kernel_size=kernel_size,
        )
        assert processor.sample_rate == sample_rate
        assert processor.min_interval_len == min_interval_len
        assert processor.erode == erode
        assert processor.kernel_size == kernel_size

    @staticmethod
    def test_inherits_post_processor() -> None:
        """MEBinPostProcessor should be a subclass of PostProcessor."""
        from anomalib.post_processing import PostProcessor

        processor = MEBinPostProcessor()
        assert isinstance(processor, PostProcessor)

    @staticmethod
    def test_forward_produces_valid_output() -> None:
        """Forward pass should produce InferenceBatch with valid fields."""
        processor = MEBinPostProcessor(sample_rate=8, min_interval_len=2)
        predictions = InferenceBatch(
            pred_score=torch.tensor([0.8]),
            anomaly_map=torch.rand(1, 1, 32, 32),
        )
        result = processor.forward(predictions)
        assert isinstance(result, InferenceBatch)
        assert result.pred_score is not None
        assert result.anomaly_map is not None
        assert result.pred_mask is not None
        assert result.pred_label is not None
        # pred_mask should be boolean
        assert result.pred_mask.dtype == torch.bool

    @staticmethod
    def test_forward_without_pred_score() -> None:
        """Forward should derive pred_score from anomaly_map if not provided."""
        processor = MEBinPostProcessor(sample_rate=8, min_interval_len=2)
        anomaly_map = torch.rand(2, 1, 16, 16)
        predictions = InferenceBatch(anomaly_map=anomaly_map)
        result = processor.forward(predictions)
        assert result.pred_score is not None

    @staticmethod
    def test_forward_raises_on_missing_inputs() -> None:
        """Forward should raise ValueError if both pred_score and anomaly_map are None."""
        processor = MEBinPostProcessor()
        predictions = InferenceBatch()
        with pytest.raises(ValueError, match="At least one of"):
            processor.forward(predictions)

    @staticmethod
    def test_forward_no_thresholding() -> None:
        """When thresholding is disabled, pred_mask and pred_label should be None."""
        processor = MEBinPostProcessor(enable_thresholding=False)
        predictions = InferenceBatch(
            pred_score=torch.tensor([0.5]),
            anomaly_map=torch.rand(1, 1, 16, 16),
        )
        result = processor.forward(predictions)
        assert result.pred_mask is None
        assert result.pred_label is None

    @staticmethod
    def test_post_process_batch() -> None:
        """post_process_batch should modify batch in-place with MEBin masks."""
        processor = MEBinPostProcessor(sample_rate=8, min_interval_len=2)
        batch = ImageBatch(
            image=torch.rand(2, 3, 32, 32),
            anomaly_map=torch.rand(2, 1, 32, 32),
            pred_score=torch.tensor([0.3, 0.8]),
        )
        processor.post_process_batch(batch)
        assert batch.pred_mask is not None
        assert batch.pred_label is not None
        # pred_mask should be binary
        unique = batch.pred_mask.unique()
        assert torch.all((unique == 0) | (unique == 1))

    @staticmethod
    def test_post_process_batch_with_normalization() -> None:
        """post_process_batch should normalize when stats are available."""
        processor = MEBinPostProcessor(sample_rate=8, min_interval_len=2)
        # Set normalization stats
        processor.pixel_min.copy_(torch.tensor(0.0))
        processor.pixel_max.copy_(torch.tensor(1.0))
        processor.image_min.copy_(torch.tensor(0.0))
        processor.image_max.copy_(torch.tensor(1.0))
        processor._image_threshold.copy_(torch.tensor(0.5))  # noqa: SLF001
        processor._pixel_threshold.copy_(torch.tensor(0.5))  # noqa: SLF001

        batch = ImageBatch(
            image=torch.rand(2, 3, 32, 32),
            anomaly_map=torch.rand(2, 1, 32, 32),
            pred_score=torch.tensor([0.3, 0.8]),
        )
        processor.post_process_batch(batch)
        # After normalization, anomaly_map should be in [0, 1]
        assert batch.anomaly_map.min() >= 0.0
        assert batch.anomaly_map.max() <= 1.0

    @staticmethod
    def test_on_test_batch_end_calls_post_process() -> None:
        """on_test_batch_end should trigger post_process_batch."""
        processor = MEBinPostProcessor(sample_rate=8, min_interval_len=2)
        batch = ImageBatch(
            image=torch.rand(1, 3, 16, 16),
            anomaly_map=torch.rand(1, 1, 16, 16),
            pred_score=torch.tensor([0.5]),
        )
        # on_test_batch_end delegates to post_process_batch
        processor.on_test_batch_end(None, None, batch)
        assert batch.pred_mask is not None

    @staticmethod
    def test_forward_pred_score_only_with_normalization() -> None:
        """forward() with pred_score only and anomaly_map=None should not crash.

        Regression test: when anomaly_map is None but normalization is enabled,
        the normalization step must not call _normalize(None, ...).
        """
        processor = MEBinPostProcessor()
        # Set normalization stats so normalization is actually attempted.
        processor.image_min.copy_(torch.tensor(0.0))
        processor.image_max.copy_(torch.tensor(1.0))
        processor._image_threshold.copy_(torch.tensor(0.5))  # noqa: SLF001

        predictions = InferenceBatch(pred_score=torch.tensor([0.3, 0.8]))
        result = processor.forward(predictions)
        assert result.pred_score is not None
        assert result.anomaly_map is None
        assert result.pred_mask is None  # No anomaly_map → no mask
