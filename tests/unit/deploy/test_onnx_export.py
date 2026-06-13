# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for ONNX export helpers."""

import pytest

from anomalib.models.components.base.export_mixin import _disable_pos_embed_antialiasing

timm_vit = pytest.importorskip("timm.models.vision_transformer")


class TestDisablePosEmbedAntialiasing:
    """Tests for the ``_disable_pos_embed_antialiasing`` context manager."""

    @staticmethod
    def test_swaps_and_restores() -> None:
        """The resample function is swapped inside the context and restored on exit."""
        original = timm_vit.resample_abs_pos_embed
        with _disable_pos_embed_antialiasing():
            assert timm_vit.resample_abs_pos_embed is not original
        assert timm_vit.resample_abs_pos_embed is original

    @staticmethod
    def test_restores_on_exception() -> None:
        """The original reference is restored even if the body raises."""
        original = timm_vit.resample_abs_pos_embed
        msg = "boom"
        with pytest.raises(RuntimeError, match=msg), _disable_pos_embed_antialiasing():
            raise RuntimeError(msg)
        assert timm_vit.resample_abs_pos_embed is original

    @staticmethod
    def test_forces_antialias_false(monkeypatch: pytest.MonkeyPatch) -> None:
        """The wrapper forwards ``antialias=False`` to the underlying resample call."""
        captured: dict[str, object] = {}

        def fake_resample(*_args: object, **kwargs: object) -> str:
            captured.update(kwargs)
            return "resampled"

        monkeypatch.setattr(timm_vit, "resample_abs_pos_embed", fake_resample)

        with _disable_pos_embed_antialiasing():
            result = timm_vit.resample_abs_pos_embed("posemb", antialias=True)

        assert result == "resampled"
        assert captured["antialias"] is False
