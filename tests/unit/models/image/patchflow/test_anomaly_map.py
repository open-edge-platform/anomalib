# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PatchFlow anomaly map generator."""

import torch
from omegaconf import ListConfig

from anomalib.models.image.patchflow.anomaly_map import AnomalyMapGenerator


def test_output_shape() -> None:
    """Test that anomaly map has the expected shape."""
    generator = AnomalyMapGenerator(input_size=(224, 224))
    hidden_variables = torch.randn(2, 64, 28, 28)

    anomaly_map = generator(hidden_variables)

    assert anomaly_map.shape == (2, 1, 224, 224)


def test_no_nan() -> None:
    """Test that the anomaly map contains no NaN values."""
    generator = AnomalyMapGenerator(input_size=(128, 128))
    hidden_variables = torch.randn(4, 32, 16, 16)

    anomaly_map = generator(hidden_variables)

    assert not torch.isnan(anomaly_map).any()


def test_tuple_and_listconfig() -> None:
    """Test that both tuple and ListConfig inputs for input_size work."""
    hidden_variables = torch.randn(1, 16, 8, 8)

    # tuple input
    gen_tuple = AnomalyMapGenerator(input_size=(64, 64))
    assert gen_tuple(hidden_variables).shape == (1, 1, 64, 64)

    # ListConfig input (e.g. from OmegaConf/Hydra configs)
    gen_listconfig = AnomalyMapGenerator(input_size=ListConfig([64, 64]))
    assert gen_listconfig(hidden_variables).shape == (1, 1, 64, 64)
