# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline feature extractor for UniADet algorithm and lifecycle tests."""

import pytest
import torch
from torch import nn
from torch.nn import functional

from anomalib.models.image.uniadet.backbones import HierarchicalFeatures


class TinyBackbone(nn.Module):
    """Deterministic, frozen test encoder; not a substitute for real-backbone benchmarks."""

    def __init__(self, backbone: str, layers: tuple[int, ...], pretrained: bool, image_size: int | None) -> None:
        super().__init__()
        del backbone, pretrained
        self.layers = layers
        self.feature_dim = 3
        self.image_size = image_size or 28
        self.grid_size = 4
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, images: torch.Tensor) -> HierarchicalFeatures:
        """Return deterministic image and patch features."""
        patches = functional.adaptive_avg_pool2d(images, (4, 4)).flatten(2).transpose(1, 2) * self.scale
        return HierarchicalFeatures(
            [patches.mean(dim=1) for _ in self.layers],
            [patches for _ in self.layers],
        )


@pytest.fixture
def tiny_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace only the encoder, keeping head, memory, loss, lifecycle, and export real."""
    monkeypatch.setattr("anomalib.models.image.uniadet.torch_model.UniADetBackbone", TinyBackbone)
