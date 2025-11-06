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
