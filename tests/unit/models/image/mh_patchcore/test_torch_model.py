# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the MH-PatchCore embedding pipeline."""

from typing import cast

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch
from torch import nn

from anomalib.models.image.mh_patchcore import torch_model
from anomalib.models.image.mh_patchcore.torch_model import MHPatchcoreModel


class MockFeatureExtractor(nn.Module):
    """Deterministic feature extractor used by the embedding tests."""

    def __init__(self, backbone: str, layers: tuple[str, ...], pre_trained: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.pre_trained = pre_trained

    def forward(self, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: PLR6301
        """Return deterministic feature maps for both configured layers."""
        batch_size = input_tensor.shape[0]
        layer2 = torch.arange(batch_size * 12, dtype=input_tensor.dtype, device=input_tensor.device)
        layer3 = torch.arange(batch_size * 4, dtype=input_tensor.dtype, device=input_tensor.device)
        return {
            "layer2": layer2.reshape(batch_size, 1, 3, 4),
            "layer3": layer3.reshape(batch_size, 1, 2, 2),
        }


@pytest.fixture
def model(monkeypatch: MonkeyPatch) -> MHPatchcoreModel:
    """Create an MH-PatchCore model without constructing a real backbone."""
    monkeypatch.setattr(torch_model, "TimmFeatureExtractor", MockFeatureExtractor)
    return MHPatchcoreModel()


def test_patchify_preserves_grid_and_order(model: MHPatchcoreModel) -> None:
    """Patch extraction should preserve batch-major and row-major ordering."""
    features = torch.tensor(
        [
            [[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]],
            [[[100.0, 101.0, 102.0], [110.0, 111.0, 112.0]]],
        ],
    )

    patches, grid = model._patchify(features)  # noqa: SLF001

    assert grid == (2, 3)
    assert patches.shape == (2, 6, 1, 3, 3)
    torch.testing.assert_close(patches[:, :, 0, 1, 1], features.flatten(start_dim=2).squeeze(1))
    torch.testing.assert_close(
        patches[0, 0, 0],
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 10.0, 11.0]]),
    )


def test_align_patches_preserves_patch_axes() -> None:
    """Spatial alignment should resize only the patch-grid dimensions."""
    grid_values = torch.tensor([0.0, 2.0, 4.0, 6.0]).reshape(1, 4, 1, 1, 1)
    channel_offsets = torch.tensor([0.0, 100.0]).reshape(1, 1, 2, 1, 1)
    patch_offsets = torch.arange(9, dtype=torch.float32).reshape(1, 1, 1, 3, 3)
    patches = grid_values + channel_offsets + patch_offsets

    aligned = MHPatchcoreModel._align_patches(  # noqa: SLF001
        patches,
        source_grid=(2, 2),
        target_grid=(3, 3),
    )

    expected_grid = torch.tensor([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]])
    assert aligned.shape == (1, 9, 2, 3, 3)
    torch.testing.assert_close(aligned[0, :, 0, 0, 0].reshape(3, 3), expected_grid)
    torch.testing.assert_close(aligned[0, :, 1, 2, 2].reshape(3, 3), expected_grid + 108)


def test_mapping_and_aggregation_preserve_layer_order() -> None:
    """Mapped features should retain configured layer order at dimension 1024."""
    first_layer = torch.full((2, 1, 3, 3), 2.0)
    second_layer = torch.full((2, 1, 3, 3), 6.0)
    mapped = torch.stack(
        [
            MHPatchcoreModel._map_features(first_layer),  # noqa: SLF001
            MHPatchcoreModel._map_features(second_layer),  # noqa: SLF001
        ],
        dim=1,
    )

    embedding = MHPatchcoreModel._aggregate_features(mapped)  # noqa: SLF001

    assert mapped.shape == (2, 2, 1024)
    assert embedding.shape == (2, 1024)
    torch.testing.assert_close(embedding[:, :512], torch.full((2, 512), 2.0))
    torch.testing.assert_close(embedding[:, 512:], torch.full((2, 512), 6.0))


def test_forward_uses_mocked_feature_extractor(model: MHPatchcoreModel) -> None:
    """Forward should combine both feature layers without downloading weights."""
    extractor = cast("MockFeatureExtractor", model.feature_extractor)
    output = model(torch.zeros(2, 3, 8, 8))

    assert extractor.backbone == "wide_resnet50_2.tv2_in1k"
    assert extractor.layers == ("layer2", "layer3")
    assert extractor.pre_trained is True
    assert output.shape == (24, 1024)
    assert torch.isfinite(output).all()
