# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PatchFlow anomaly map generator."""

import torch

from anomalib.models.image.patchflow.anomaly_map import AnomalyMapGenerator


class TestAnomalyMapGenerator:
    """Tests for AnomalyMapGenerator."""

    def test_output_shape(self) -> None:
        """Test that anomaly map has the expected shape."""
        generator = AnomalyMapGenerator(input_size=(224, 224))
        hidden_variables = torch.randn(2, 64, 28, 28)

        anomaly_map = generator(hidden_variables)

        assert anomaly_map.shape == (2, 1, 224, 224)

    def test_no_nan(self) -> None:
        """Test that the anomaly map contains no NaN values."""
        generator = AnomalyMapGenerator(input_size=(128, 128))
        hidden_variables = torch.randn(4, 32, 16, 16)

        anomaly_map = generator(hidden_variables)

        assert not torch.isnan(anomaly_map).any()

    def test_tuple_and_listconfig(self) -> None:
        """Test that both tuple and list inputs for input_size work."""
        generator = AnomalyMapGenerator(input_size=(64, 64))
        hidden_variables = torch.randn(1, 16, 8, 8)

        anomaly_map = generator(hidden_variables)

        assert anomaly_map.shape == (1, 1, 64, 64)
