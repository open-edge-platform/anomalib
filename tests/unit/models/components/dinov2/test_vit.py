# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DINOv2 ViT / Loader."""

from __future__ import annotations

import pytest
import torch

from anomalib.models.components.dinov2.vision_transformer import (
    DinoVisionTransformer,
    vit_base,
    vit_large,
    vit_small,
)


@pytest.fixture()
def tiny_vit() -> DinoVisionTransformer:
    """Return a very small ViT model for unit testing."""
    return DinoVisionTransformer(
        img_size=32,
        patch_size=8,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )


@pytest.fixture()
def tiny_input() -> torch.Tensor:
    """Return a small dummy input tensor."""
    return torch.randn(2, 3, 32, 32)  # (B=2, C=3, H=W=32)


def test_model_initializes(tiny_vit: DinoVisionTransformer) -> None:
    """Model constructs and exposes expected attributes."""
    m: DinoVisionTransformer = tiny_vit

    assert m.embed_dim == 64
    assert m.patch_size == 8
    assert m.n_blocks == 2
    assert hasattr(m, "patch_embed")
    assert hasattr(m, "cls_token")
    assert hasattr(m, "pos_embed")
    assert hasattr(m, "blocks")


def test_patch_embedding_shape(
    tiny_vit: DinoVisionTransformer,
    tiny_input: torch.Tensor,
) -> None:
    """Patch embedding output has correct (B, N, C) shape."""
    patches: torch.Tensor = tiny_vit.patch_embed(tiny_input)
    b, n, c = patches.shape

    assert b == 2
    assert n == 16  # 32x32 with patch_size=8 → 4x4 → 16 patches
    assert tiny_vit.embed_dim == c


def test_prepare_tokens_output_shape(
    tiny_vit: DinoVisionTransformer,
    tiny_input: torch.Tensor,
) -> None:
    """prepare_tokens_with_masks adds CLS and keeps correct embedding dims."""
    tokens: torch.Tensor = tiny_vit.prepare_tokens_with_masks(tiny_input)

    expected_tokens: int = 1 + tiny_vit.patch_embed.num_patches
    assert tokens.shape == (2, expected_tokens, tiny_vit.embed_dim)


def test_forward_features_training_output_shapes(
    tiny_vit: DinoVisionTransformer,
    tiny_input: torch.Tensor,
) -> None:
    """forward(is_training=True) returns a dict with expected shapes."""
    out: dict[str, torch.Tensor | None] = tiny_vit(tiny_input, is_training=True)  # type: ignore[assignment]

    assert isinstance(out, dict)
    assert out["x_norm_clstoken"] is not None
    assert out["x_norm_patchtokens"] is not None

    cls: torch.Tensor = out["x_norm_clstoken"]  # type: ignore[assignment]
    patches: torch.Tensor = out["x_norm_patchtokens"]  # type: ignore[assignment]

    assert cls.shape == (2, tiny_vit.embed_dim)
    assert patches.shape[1] == tiny_vit.patch_embed.num_patches


def test_forward_inference_output_shape(
    tiny_vit: DinoVisionTransformer,
    tiny_input: torch.Tensor,
) -> None:
    """Inference mode returns class-token output only."""
    out: torch.Tensor = tiny_vit(tiny_input)  # default is is_training=False

    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, tiny_vit.embed_dim)


def test_get_intermediate_layers_shapes(
    tiny_vit: DinoVisionTransformer,
    tiny_input: torch.Tensor,
) -> None:
    """Intermediate layer extraction returns tensors shaped (B, tokens, C)."""
    feats: tuple[torch.Tensor, ...] = tiny_vit.get_intermediate_layers(
        tiny_input,
        n=1,
    )

    assert len(feats) == 1

    f: torch.Tensor = feats[0]
    assert f.shape[0] == 2  # batch
    assert f.shape[2] == tiny_vit.embed_dim


@pytest.mark.parametrize(
    "factory",
    [vit_small, vit_base, vit_large],
)
def test_vit_factories_create_models(factory) -> None:  # noqa: ANN001
    """vit_small/base/large should return valid models."""
    model: DinoVisionTransformer = factory()

    assert isinstance(model, DinoVisionTransformer)
    assert model.embed_dim > 0
    assert model.n_blocks > 0
    assert model.num_heads > 0
