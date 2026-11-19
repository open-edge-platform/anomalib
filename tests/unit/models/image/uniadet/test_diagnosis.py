# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline invariants for the FP32 component-isolation experiments."""

import json
import math
from pathlib import Path

import pytest
import torch
from tools.uniadet.diagnose import main, scoring_variants
from torch import nn

from anomalib.models.image.uniadet.backbones import HierarchicalFeatures
from anomalib.models.image.uniadet.torch_model import UniADetModel


@pytest.mark.usefixtures("tiny_backbone")
@pytest.mark.parametrize("shots", [0, 2])
def test_diagnostic_baseline_preserves_detector(shots: int) -> None:
    """Ablation comparisons must start from exactly the production scores and maps."""
    torch.manual_seed(10)
    model = UniADetModel(pretrained=False, image_size=28)
    model.head_trained.fill_(value=True)
    if shots:
        model.set_reference_images(torch.rand(shots, 3, 28, 28))
    features = model.backbone(torch.rand(3, 3, 28, 28))
    expected = model.predict_from_features(features)
    actual = scoring_variants(model, features)["baseline"]
    torch.testing.assert_close(actual.pred_score, expected.pred_score)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map)


@pytest.mark.usefixtures("tiny_backbone")
def test_paper_fusion_has_hand_computed_nearest_distance() -> None:
    """Orthogonal unit vectors isolate nearest-neighbor reduction and both fusion weights."""
    model = UniADetModel(pretrained=False, layers=(1, 2), image_size=28, temperature=1)
    model.head_trained.fill_(value=True)
    model.backbone.grid_size = 2
    model.blur = nn.Identity()
    with torch.no_grad():
        model.head.image_prototypes.copy_(torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]] * 2))
        model.head.pixel_prototypes.copy_(torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]] * 2))
    image = torch.tensor([[1.0, 0.0, 0.0]])
    patches = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]])
    features = HierarchicalFeatures([image, image], [patches, patches])
    model.memory_bank = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]] * 2)
    distances = model._reference_distances(features)  # noqa: SLF001
    torch.testing.assert_close(distances, torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]]))
    prediction = model.predict_from_features(features)
    probability = 1 / (1 + math.exp(-1))
    expected_corners = torch.tensor([
        [
            [
                [
                    (1 - probability) / 2,
                    probability / 2,
                ],
                [
                    (1 + probability) / 2,
                    (2 - probability) / 2,
                ],
            ],
        ],
    ])
    torch.testing.assert_close(prediction.anomaly_map[:, :, ::27, ::27], expected_corners)
    assert prediction.pred_score.item() == pytest.approx((1 - probability) / 2 + (1 + probability) / 4)


@pytest.mark.usefixtures("tiny_backbone")
def test_published_scoring_probe_uses_declared_recalibration() -> None:
    """Constant maps isolate the executable's sigmoid and fusion from interpolation and smoothing."""
    model = UniADetModel(pretrained=False, layers=(1,), image_size=28, temperature=1)
    with torch.no_grad():
        model.head.image_prototypes.copy_(torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]))
        model.head.pixel_prototypes.copy_(model.head.image_prototypes)
    image = torch.tensor([[1.0, 0.0, 0.0]])
    patches = image.unsqueeze(1).expand(1, 16, 3)
    features = HierarchicalFeatures([image], [patches])
    model.memory_bank = torch.tensor([[[0.0, 1.0, 0.0]]])
    predictions = scoring_variants(model, features)
    probability = 1 / (1 + math.exp(1))
    zero_map = 1 / (1 + math.exp(-15 * (probability - 0.4)))
    reference_map = 1 / (1 + math.exp(-15 * (1 - 0.3)))
    expected_map = 0.2 * zero_map + 0.8 * reference_map
    expected_score = 0.5 * probability + 0.5 * expected_map
    torch.testing.assert_close(
        predictions["published_executable"].anomaly_map,
        torch.full((1, 1, 28, 28), expected_map),
    )
    assert predictions["published_executable"].pred_score.item() == pytest.approx(expected_score, abs=1e-6)


@pytest.mark.parametrize(
    ("source", "category", "message"),
    [
        ("mvtec", "bottle", "must differ"),
        ("visa", "../outside", "Categories must belong"),
    ],
)
def test_diagnostic_rejects_invalid_protocol_before_loading_encoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    category: str,
    message: str,
) -> None:
    """Same-domain supervision and invalid categories fail without a model download."""
    (tmp_path / "training.json").write_text(json.dumps({"source_dataset": source}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnose",
            "--weights",
            str(tmp_path / "head.pt"),
            "--root",
            str(tmp_path),
            "--output",
            str(tmp_path / "output"),
            "--categories",
            category,
        ],
    )
    with pytest.raises(ValueError, match=message):
        main()
