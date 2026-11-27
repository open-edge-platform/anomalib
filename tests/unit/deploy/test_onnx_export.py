# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for backbone-local positional interpolation and its deployment round trip."""

from __future__ import annotations

import copy
import functools
import inspect
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
import timm.models.vision_transformer as vit
import torch

from anomalib.models.components.feature_extractors import TimmFeatureExtractor
from anomalib.models.image.foundad.components.encoder_loader import load_encoder

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from pytest_mock import MockerFixture


@pytest.fixture
def tiny_vit(mocker: MockerFixture) -> Iterator[vit.VisionTransformer]:
    """Provide an offline transformer and isolate any legacy global patch."""
    mocker.patch.object(vit, "resample_abs_pos_embed", inspect.unwrap(vit.resample_abs_pos_embed))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield vit.VisionTransformer(
            img_size=32,
            patch_size=4,
            embed_dim=24,
            depth=2,
            num_heads=3,
            reg_tokens=2,
            dynamic_img_size=True,
        )


@pytest.mark.parametrize("legacy_layout", [False, True])
@pytest.mark.parametrize("wrapper", ["extractor", "foundad"])
def test_transformer_cold_load_preserves_interpolation(
    mocker: MockerFixture,
    tmp_path: Path,
    tiny_vit: vit.VisionTransformer,
    legacy_layout: bool,
    wrapper: str,
) -> None:
    """A fresh process restores both new and older full-object exports without a constructor."""
    mocker.patch("timm.create_model", return_value=tiny_vit)
    model = (
        TimmFeatureExtractor("vit_tiny", ["blocks.0"], pre_trained=False, output_fmt="NLC")
        if wrapper == "extractor"
        else load_encoder("dinov2_vit_small_14")
    ).eval()
    method = "forward" if wrapper == "extractor" else "get_intermediate_layers"
    images = torch.rand(1, 3, 24, 40)
    with torch.no_grad():
        expected = getattr(model, method)(images)
    if legacy_layout:
        tiny_vit.__dict__.pop("_pos_embed", None)
    payload = tmp_path / "extractor.pt"
    # Full-object loading is intentional; only this test's locally generated payload is loaded.
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    torch.save({"model": model, "method": method, "images": images, "expected": expected}, payload)

    script = """
import sys
import torch
torch.set_num_threads(2)
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
with torch.no_grad():
    actual = getattr(payload["model"], payload["method"])(payload["images"])
torch.testing.assert_close(actual, payload["expected"], rtol=1e-6, atol=1e-6)
"""
    result = subprocess.run(  # noqa: S603 - fixed script and locally generated payload
        [sys.executable, "-c", script, str(payload)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_transformer_does_not_change_other_timm_models(
    mocker: MockerFixture,
    tiny_vit: vit.VisionTransformer,
) -> None:
    """Creating an extractor leaves timm's shared resampler and other instances untouched."""
    original = vit.resample_abs_pos_embed
    unrelated = copy.deepcopy(tiny_vit).eval()
    images = torch.rand(1, 3, 24, 40)
    with torch.no_grad():
        expected = unrelated.forward_features(images)
    mocker.patch("timm.create_model", return_value=tiny_vit)

    TimmFeatureExtractor("vit_tiny", ["blocks.0"], pre_trained=False, output_fmt="NLC")

    assert vit.resample_abs_pos_embed is original
    with torch.no_grad():
        torch.testing.assert_close(unrelated.forward_features(images), expected, rtol=0, atol=0)


@torch.enable_grad()
def test_transformer_preserves_no_antialias_features_and_gradients(
    mocker: MockerFixture,
    tiny_vit: vit.VisionTransformer,
) -> None:
    """Local interpolation matches the previous policy for features and gradients."""
    reference = copy.deepcopy(tiny_vit).eval()
    mocker.patch("timm.create_model", return_value=tiny_vit)
    extractor = TimmFeatureExtractor(
        "vit_tiny",
        ["blocks.0"],
        pre_trained=False,
        requires_grad=True,
        output_fmt="NLC",
        return_class_token=True,
        norm=False,
    ).eval()
    images = torch.rand(1, 3, 24, 40)
    actual = extractor(images)["blocks.0"]
    mocker.patch.object(
        vit,
        "resample_abs_pos_embed",
        functools.partial(inspect.unwrap(vit.resample_abs_pos_embed), antialias=False),
    )
    patches, prefix = reference.forward_intermediates(
        images,
        indices=[0],
        return_prefix_tokens=True,
        output_fmt="NLC",
        intermediates_only=True,
    )[0]
    expected = torch.cat([prefix, patches], dim=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(tiny_vit.pos_embed.grad, reference.pos_embed.grad, rtol=0, atol=0)


def test_transformer_exports_dynamic_batches(
    mocker: MockerFixture,
    tmp_path: Path,
    tiny_vit: vit.VisionTransformer,
) -> None:
    """The serialized interpolation callable remains traceable with real variable-batch inference."""
    pytest.importorskip("onnx")
    ov = pytest.importorskip("openvino")
    mocker.patch("timm.create_model", return_value=tiny_vit)
    extractor = TimmFeatureExtractor("vit_tiny", ["blocks.0"], pre_trained=False, output_fmt="NLC").eval()
    images = torch.rand(3, 3, 24, 40)
    path = tmp_path / "extractor.onnx"
    torch.onnx.export(
        extractor,
        (images[:2],),
        path,
        dynamo=True,
        input_names=["images"],
        output_names=["features"],
        dynamic_shapes={"inputs": {0: torch.export.Dim("batch", min=1, max=3)}},
    )
    compiled = ov.Core().compile_model(
        path,
        "CPU",
        {"INFERENCE_PRECISION_HINT": "f32", "INFERENCE_NUM_THREADS": 2},
    )
    for batch in (1, 2, 3):
        actual = compiled(images[:batch].numpy())[0]
        expected = extractor(images[:batch])["blocks.0"]
        torch.testing.assert_close(torch.from_numpy(actual), expected, rtol=1e-4, atol=1e-5)
