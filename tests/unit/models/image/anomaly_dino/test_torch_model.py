# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the AnomalyDINO torch model."""

import numpy as np
import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch

from anomalib.models.image.anomaly_dino.torch_model import AnomalyDINOModel


class TestAnomalyDINOModel:
    """Test the AnomalyDINO torch model."""

    @staticmethod
    def test_initialization_defaults() -> None:
        """Test initialization with default arguments."""
        model = AnomalyDINOModel()
        assert model.encoder_name.startswith("dinov2")
        assert model.memory_bank.numel() == 0

    @staticmethod
    def test_invalid_encoder_name_raises() -> None:
        """Test that invalid encoder names raise an error."""
        with pytest.raises(ValueError, match="Encoder name must start with 'dinov2', got 'resnet50'"):
            _ = AnomalyDINOModel(encoder_name="resnet50")

    @staticmethod
    def test_fit_raises_without_embeddings() -> None:
        """Test that fit raises when no embeddings have been collected."""
        model = AnomalyDINOModel()
        with pytest.raises(ValueError, match="No embeddings collected"):
            model.fit()

    @staticmethod
    def test_forward_train_adds_embeddings(monkeypatch: MonkeyPatch) -> None:
        """Test training mode collects embeddings into store."""
        model = AnomalyDINOModel()
        model.train()

        fake_features = torch.randn(2, 8, 128)
        monkeypatch.setattr(model, "extract_features", lambda _: fake_features)

        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        assert torch.is_tensor(output)
        assert output.requires_grad
        assert len(model.embedding_store) == 1
        assert model.embedding_store[0].ndim == 2

    @staticmethod
    def test_forward_eval_raises_with_empty_memory_bank(monkeypatch: MonkeyPatch) -> None:
        """Test that inference raises an error when memory bank is empty."""
        model = AnomalyDINOModel()
        model.eval()

        fake_features = torch.randn(1, 16, 64)
        monkeypatch.setattr(model, "extract_features", lambda _: fake_features)
        model.register_buffer("memory_bank", torch.empty(0, 64))

        x = torch.randn(1, 3, 224, 224)
        with pytest.raises(RuntimeError, match="Memory bank is empty"):
            _ = model(x)

    @staticmethod
    def test_compute_background_masks_runs() -> None:
        """Test that background mask computation produces boolean masks."""
        b, h, w, d = 2, 8, 8, 16
        features = np.random.randn(b, h * w, d).astype(np.float32)  # noqa: NPY002
        masks = AnomalyDINOModel.compute_background_masks(features, (h, w))
        assert masks.shape == (b, h * w)
        assert masks.dtype == bool

    @staticmethod
    def test_mean_top1p_computation() -> None:
        """Test that mean_top1p returns expected shape and value."""
        distances = torch.arange(0, 100, dtype=torch.float32).view(1, -1)
        result = AnomalyDINOModel.mean_top1p(distances)
        assert result.shape == (1, 1)
        assert torch.allclose(result, torch.tensor([[99.0]]))

    @staticmethod
    def test_forward_half_precision_eval(monkeypatch: MonkeyPatch) -> None:
        """Test inference in half precision (float16) using matmul cosine distance."""
        model = AnomalyDINOModel().half()
        model.eval()

        fake_features = torch.randn(1, 16, 64, dtype=torch.float16)
        monkeypatch.setattr(model, "extract_features", lambda _: fake_features)
        monkeypatch.setattr(model.anomaly_map_generator, "__call__", lambda x, __: x)

        model.register_buffer("memory_bank", torch.randn(16, 64, dtype=torch.float16))
        x = torch.randn(1, 3, 224, 224, dtype=torch.float16)
        out = model(x)

        assert hasattr(out, "pred_score")
        assert out.pred_score.shape == (1, 1)
        # outputs should be float16-safe with matmul
        assert out.pred_score.dtype == torch.float16
