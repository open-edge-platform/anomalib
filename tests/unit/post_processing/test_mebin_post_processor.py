# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test the MEBinPostProcessor class."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from anomalib.data import InferenceBatch
from anomalib.post_processing import MEBinPostProcessor


class TestMEBinPostProcessor:
    """Test the MEBinPostProcessor class."""

    @staticmethod
    def test_initialization_default_params() -> None:
        """Test MEBinPostProcessor initialization with default parameters."""
        processor = MEBinPostProcessor()

        assert processor.sample_rate == 4
        assert processor.min_interval_len == 4
        assert processor.erode is True

    @staticmethod
    @pytest.mark.parametrize(
        ("sample_rate", "min_interval_len", "erode"),
        [
            (2, 3, True),
            (8, 6, False),
            (1, 1, True),
        ],
    )
    def test_initialization_custom_params(
        sample_rate: int,
        min_interval_len: int,
        erode: bool,
    ) -> None:
        """Test MEBinPostProcessor initialization with custom parameters."""
        processor = MEBinPostProcessor(
            sample_rate=sample_rate,
            min_interval_len=min_interval_len,
            erode=erode,
        )

        assert processor.sample_rate == sample_rate
        assert processor.min_interval_len == min_interval_len
        assert processor.erode == erode

    @staticmethod
    @patch("anomalib.post_processing.mebin_post_processor.MEBin")
    def test_forward_single_anomaly_map(mock_mebin: MagicMock) -> None:
        """Test forward method with single anomaly map."""
        # Setup mock
        mock_mebin_instance = Mock()
        mock_mebin_instance.binarize_anomaly_maps.return_value = (
            [np.array([[0, 0], [1, 1]], dtype=np.uint8)],
            [0.5],
        )
        mock_mebin.return_value = mock_mebin_instance

        # Create test data
        anomaly_map = torch.rand(1, 1, 4, 4)
        predictions = InferenceBatch(
            pred_score=torch.tensor([0.8]),
            pred_label=torch.tensor([1]),
            anomaly_map=anomaly_map,
            pred_mask=None,
        )

        # Test forward pass
        processor = MEBinPostProcessor()
        result = processor.forward(predictions)

        # Verify results
        assert isinstance(result, InferenceBatch)
        assert result.pred_mask is not None
        assert result.pred_mask.shape == (1, 2, 2)
        assert result.pred_mask.dtype == anomaly_map.dtype

    @staticmethod
    @patch("anomalib.post_processing.mebin_post_processor.MEBin")
    def test_forward_batch_anomaly_maps(mock_mebin: MagicMock) -> None:
        """Test forward method with batch of anomaly maps."""
        # Setup mock
        mock_mebin_instance = Mock()
        mock_mebin_instance.binarize_anomaly_maps.return_value = (
            [
                np.array([[0, 0], [1, 1]], dtype=np.uint8),
                np.array([[1, 0], [0, 1]], dtype=np.uint8),
            ],
            [0.5, 0.6],
        )
        mock_mebin.return_value = mock_mebin_instance

        # Create test data
        anomaly_maps = torch.rand(2, 1, 4, 4)
        predictions = InferenceBatch(
            pred_score=torch.tensor([0.8, 0.9]),
            pred_label=torch.tensor([1, 1]),
            anomaly_map=anomaly_maps,
            pred_mask=None,
        )

        # Test forward pass
        processor = MEBinPostProcessor()
        result = processor.forward(predictions)

        # Verify results
        assert isinstance(result, InferenceBatch)
        assert result.pred_mask.shape == (2, 2, 2)

    @staticmethod
    @patch("anomalib.post_processing.mebin_post_processor.MEBin")
    def test_forward_normalization(mock_mebin: MagicMock) -> None:
        """Test that anomaly maps are properly normalized to 0-255 range."""
        # Setup mock
        mock_mebin_instance = Mock()
        mock_mebin_instance.binarize_anomaly_maps.return_value = (
            [np.array([[0, 0], [1, 1]], dtype=np.uint8)],
            [0.5],
        )
        mock_mebin.return_value = mock_mebin_instance

        # Create test data with specific range
        anomaly_maps = torch.tensor([[[[0.0, 0.5], [1.0, 0.2]]]])
        predictions = InferenceBatch(
            pred_score=torch.tensor([0.8]),
            pred_label=torch.tensor([1]),
            anomaly_map=anomaly_maps,
            pred_mask=None,
        )

        # Test forward pass
        processor = MEBinPostProcessor()
        processor.forward(predictions)

        # Verify MEBin was called with normalized data
        mock_mebin.assert_called_once()
        call_args = mock_mebin.call_args
        anomaly_map_list = call_args[1]["anomaly_map_list"]

        # Check that the data is normalized to 0-255 range
        assert len(anomaly_map_list) == 1
        assert anomaly_map_list[0].dtype == np.uint8
        assert anomaly_map_list[0].min() >= 0
        assert anomaly_map_list[0].max() <= 255

    @staticmethod
    @patch("anomalib.post_processing.mebin_post_processor.MEBin")
    def test_forward_mebin_parameters(mock_mebin: MagicMock) -> None:
        """Test that MEBin is called with correct parameters."""
        # Setup mock
        mock_mebin_instance = Mock()
        mock_mebin_instance.binarize_anomaly_maps.return_value = (
            [np.array([[0, 0], [1, 1]], dtype=np.uint8)],
            [0.5],
        )
        mock_mebin.return_value = mock_mebin_instance

        # Create test data
        anomaly_maps = torch.rand(1, 1, 4, 4)
        predictions = InferenceBatch(
            pred_score=torch.tensor([0.8]),
            pred_label=torch.tensor([1]),
            anomaly_map=anomaly_maps,
            pred_mask=None,
        )

        # Test with custom parameters
        processor = MEBinPostProcessor(
            sample_rate=8,
            min_interval_len=6,
            erode=False,
        )
        _ = processor.forward(predictions)

        # Verify MEBin was called with correct parameters
        mock_mebin.assert_called_once_with(
            anomaly_map_list=mock_mebin.call_args[1]["anomaly_map_list"],
            sample_rate=8,
            min_interval_len=6,
            erode=False,
        )

    @staticmethod
    @patch("anomalib.post_processing.mebin_post_processor.MEBin")
    def test_forward_binary_mask_conversion(mock_mebin: MagicMock) -> None:
        """Test that binary masks are properly converted to 0/1 values."""
        # Setup mock to return masks with values > 0
        mock_mebin_instance = Mock()
        mock_mebin_instance.binarize_anomaly_maps.return_value = (
            [np.array([[0, 128], [255, 64]], dtype=np.uint8)],
            [0.5],
        )
        mock_mebin.return_value = mock_mebin_instance

        # Create test data
        anomaly_maps = torch.rand(1, 1, 2, 2)
        predictions = InferenceBatch(
            pred_score=torch.tensor([0.8]),
            pred_label=torch.tensor([1]),
            anomaly_map=anomaly_maps,
            pred_mask=None,
        )

        # Test forward pass
        processor = MEBinPostProcessor()
        result = processor.forward(predictions)

        # Verify that all values are either 0 or 1
        unique_values = torch.unique(result.pred_mask)
        assert torch.all((unique_values == 0) | (unique_values == 1))

    @staticmethod
    def test_forward_missing_anomaly_map() -> None:
        """Test that ValueError is raised when anomaly_map is None."""
        # Create test data without anomaly_map
        predictions = InferenceBatch(
            pred_score=torch.tensor([0.8]),
            pred_label=torch.tensor([1]),
            anomaly_map=None,
            pred_mask=None,
        )

        # Test forward pass should raise ValueError
        processor = MEBinPostProcessor()
        with pytest.raises(ValueError, match="Anomaly map is required for MEBin post-processing"):
            processor.forward(predictions)
