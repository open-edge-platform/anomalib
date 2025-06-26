"""Test Patchcore model compute on CPU functionality."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from types import SimpleNamespace

from anomalib.models import Patchcore
from anomalib.data import MVTecAD


class TestCpuCoresetPatchcore:
    """Test the Patchcore model with compute_coreset_on_cpu=True."""

    @staticmethod
    @pytest.mark.parametrize("compute_coreset_on_cpu", [True, False])
    def test_model_initialization_on_off_cpu(compute_coreset_on_cpu: bool) -> None:
        """Test that the Patchcore model is initialized with flag."""
        _ = Patchcore(compute_coreset_on_cpu=compute_coreset_on_cpu)

    @staticmethod
    @pytest.mark.parametrize("device_str", ["cuda", "cpu"])
    @pytest.mark.parametrize("compute_coreset_on_cpu", [True, False])
    def test_training_step_outputs_cpu_embeddings(device_str: str, compute_coreset_on_cpu: bool) -> None:
        """Test that Patchcore extract_features returns CPU tensors."""
        # on GPU
        model = Patchcore(compute_coreset_on_cpu=compute_coreset_on_cpu)
        model.to(torch.device(device_str))
        model.train()

        dummy_input = torch.rand((4, 3, 224, 224))  # Batch of 4 images
        dummy_input = dummy_input.to(torch.device(device_str))
        dummy_batch = SimpleNamespace(image=dummy_input)

        with torch.no_grad():
            model.training_step(dummy_batch)
            features = torch.vstack(model.embeddings)

        assert isinstance(features, torch.Tensor), "Expected output to be a tensor"
        if compute_coreset_on_cpu:
            assert features.device.type == "cpu", "Expected features to be on CPU"
        else:
            assert features.device.type == device_str, f"Expected features to be on {device_str}"
