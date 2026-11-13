# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the WinCLIP torch model."""

import pytest
import torch

from anomalib.models.image.winclip.torch_model import WinClipModel


class TokenMixingTransformer(torch.nn.Module):
    """Stand-in for the open_clip transformer that mixes tokens along its sequence axis, as attention does.

    It makes the test independent of the installed open_clip version. If the input layout does not match
    ``batch_first``, tokens are mixed across windows and all windows end up with the same embedding.
    """

    def __init__(self, batch_first: bool) -> None:
        super().__init__()
        self.batch_first = batch_first

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Replace each token by the mean over the sequence axis."""
        return tokens.mean(dim=1 if self.batch_first else 0, keepdim=True).expand_as(tokens)


class TestSetupWinClipModel:
    """Test the WinCLIP torch model."""

    @staticmethod
    def test_zero_shot_from_init() -> None:
        """Test WinCLIP initialization from init method in zero-shot mode."""
        model = WinClipModel(class_name="item")
        assert model.k_shot == 0
        assert getattr(model, "text_embeddings", None) is not None

    @staticmethod
    def test_zero_shot_from_setup() -> None:
        """Test WinCLIP initialization from setup method in zero-shot mode."""
        model = WinClipModel()
        model.setup(class_name="item")
        assert model.k_shot == 0
        assert getattr(model, "text_embeddings", None) is not None

    @staticmethod
    @pytest.mark.parametrize("apply_transform", [True, False])
    def test_few_shot_from_init(apply_transform: bool) -> None:
        """Test WinCLIP initialization from init in few-shot mode."""
        ref_images = torch.rand(2, 3, 240, 240)
        model = WinClipModel(class_name="item", reference_images=ref_images, apply_transform=apply_transform)
        assert model.k_shot == 2
        assert getattr(model, "text_embeddings", None) is not None
        assert getattr(model, "visual_embeddings", None) is not None

    @staticmethod
    @pytest.mark.parametrize("apply_transform", [True, False])
    def test_few_shot_from_setup(apply_transform: bool) -> None:
        """Test WinCLIP initialization from setup method in few-shot mode."""
        ref_images = torch.rand(2, 3, 240, 240)
        model = WinClipModel(apply_transform=apply_transform)
        model.setup(class_name="item", reference_images=ref_images)
        assert model.k_shot == 2
        assert getattr(model, "text_embeddings", None) is not None
        assert getattr(model, "visual_embeddings", None) is not None

    @staticmethod
    def test_window_embeddings_differ_across_windows() -> None:
        """Test that each sliding window gets its own embedding and the zero-shot anomaly map is not constant."""
        torch.manual_seed(0)
        model = WinClipModel(class_name="item")
        batch = torch.rand(1, 3, 240, 240)
        _, window_embeddings, _ = model.encode_image(batch)
        for embeddings in window_embeddings:
            assert not torch.allclose(embeddings[0, 0], embeddings[0, -1])
        assert model(batch).anomaly_map.std() > 0

    @staticmethod
    @pytest.mark.parametrize("batch_first", [True, False])
    def test_window_embeddings_respect_transformer_layout(batch_first: bool) -> None:
        """Test that windows are not mixed, for both the batch-first and the legacy transformer layouts."""
        torch.manual_seed(0)
        model = WinClipModel(class_name="item")
        model.clip.visual.transformer = TokenMixingTransformer(batch_first=batch_first)
        masks = model.masks[0]
        n_tokens = int(masks.max()) + 2  # patch tokens and the class token
        feature_map = torch.rand(1, n_tokens, model.clip.visual.ln_pre.normalized_shape[0])
        embeddings = model._get_window_embeddings(feature_map, masks)  # noqa: SLF001
        assert embeddings.shape[:2] == (1, masks.shape[1])
        assert not torch.allclose(embeddings[0, 0], embeddings[0, -1])

    @staticmethod
    def test_raises_error_when_not_initialized() -> None:
        """Test if an error is raised when trying to access un-initialized attributes."""
        model = WinClipModel()
        with pytest.raises(RuntimeError):
            _ = model.text_embeddings
        with pytest.raises(RuntimeError):
            _ = model.visual_embeddings
        with pytest.raises(RuntimeError):
            _ = model.patch_embeddings
