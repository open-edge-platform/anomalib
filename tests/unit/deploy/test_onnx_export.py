# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for ViT pos-embed antialiasing handling that enables ONNX export."""

import pytest
import torch

from anomalib.models.components.feature_extractors import TimmFeatureExtractor
from anomalib.models.components.feature_extractors.timm import _disable_pos_embed_antialiasing

timm_vit = pytest.importorskip("timm.models.vision_transformer")


class TestDisablePosEmbedAntialiasing:
    """Tests for the ``_disable_pos_embed_antialiasing`` global patch."""

    @staticmethod
    def test_installs_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
        """The timm resample function is wrapped to force ``antialias=False``."""
        captured: dict[str, object] = {}

        def fake_resample(*_args: object, **kwargs: object) -> str:
            captured.update(kwargs)
            return "resampled"

        monkeypatch.setattr(timm_vit, "resample_abs_pos_embed", fake_resample)

        _disable_pos_embed_antialiasing()

        result = timm_vit.resample_abs_pos_embed("posemb", antialias=True)
        assert result == "resampled"
        assert captured["antialias"] is False

    @staticmethod
    def test_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling the patch twice does not stack wrappers."""
        monkeypatch.setattr(timm_vit, "resample_abs_pos_embed", lambda *a, **k: (a, k))

        _disable_pos_embed_antialiasing()
        wrapped_once = timm_vit.resample_abs_pos_embed
        _disable_pos_embed_antialiasing()

        assert timm_vit.resample_abs_pos_embed is wrapped_once


class TestFeatureExtractorPosEmbedPatch:
    """The NLC ViT feature extractor disables pos-embed antialiasing for export compatibility."""

    @staticmethod
    def test_forward_resamples_without_antialiasing(monkeypatch: pytest.MonkeyPatch) -> None:
        """Building an NLC ViT extractor makes pos-embed resampling use ``antialias=False``."""
        real_resample = timm_vit.resample_abs_pos_embed
        captured: dict[str, object] = {}

        def spy_resample(*args: object, **kwargs: object) -> torch.Tensor:
            captured.update(kwargs)
            return real_resample(*args, **kwargs)

        monkeypatch.setattr(timm_vit, "resample_abs_pos_embed", spy_resample)

        model = TimmFeatureExtractor(
            backbone="vit_base_patch14_reg4_dinov2",
            layers=["blocks.2"],
            pre_trained=False,
            output_fmt="NLC",
            dynamic_img_size=True,
        )
        # Input size differs from the backbone default, forcing a pos-embed resample.
        features = model(torch.rand((1, 3, 392, 392)))

        assert captured.get("antialias") is False
        assert features["blocks.2"].shape == torch.Size((1, 28 * 28, 768))
