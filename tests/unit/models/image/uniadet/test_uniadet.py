# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for UniADet prototype learning, reference memory, and checkpoints."""

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from anomalib import LearningType
from anomalib.models import UniADet, get_model
from anomalib.models.image.uniadet.backbones import HierarchicalFeatures, UniADetBackbone
from anomalib.models.image.uniadet.head import MSDCSHead, UniADetLogits
from anomalib.models.image.uniadet.loss import UniADetLoss
from anomalib.models.image.uniadet.torch_model import UniADetModel


def test_decoupled_cosine_logits() -> None:
    """Image and patch heads must have independent decision boundaries."""
    head = MSDCSHead(2, 1, temperature=1)
    with torch.no_grad():
        head.image_prototypes.copy_(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
        head.pixel_prototypes.copy_(torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    features = HierarchicalFeatures([torch.tensor([[2.0, 0.0]])], [torch.tensor([[[3.0, 0.0]]])])
    logits = head(features, grid_size=1)
    torch.testing.assert_close(logits.image[0], torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(logits.pixel[0].flatten(1), torch.tensor([[0.0, 1.0]]))


@pytest.mark.usefixtures("tiny_backbone")
def test_auxiliary_loss_trains_both_heads() -> None:
    """Both tasks receive finite gradients while the encoder stays frozen."""
    torch.manual_seed(4)
    model = UniADetModel(pretrained=False, layers=(1, 2), image_size=28)
    images = torch.rand(2, 3, 28, 28)
    logits = model.head(model.backbone(images), model.backbone.grid_size)
    mask = torch.zeros(2, 28, 28)
    mask[1, 10:20, 10:20] = 1
    loss = UniADetLoss()(logits, torch.tensor([0, 1]), mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head.image_prototypes.grad.abs().sum() > 0
    assert model.head.pixel_prototypes.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_head_learns_supervised_prototypes() -> None:
    """The configured Adam steps must rotate both prototype tasks enough to learn auxiliary classes."""
    torch.manual_seed(7)
    head = MSDCSHead(64, 1)
    tokens = torch.eye(64)[:2]
    features = HierarchicalFeatures([tokens], [tokens.unsqueeze(1)])
    labels = torch.tensor([0, 1])
    masks = labels.reshape(2, 1, 1)
    loss_function = UniADetLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=0.001, betas=(0.5, 0.999))
    for _ in range(25):
        optimizer.zero_grad()
        loss = loss_function(head(features, 1), labels, masks)
        loss.backward()
        optimizer.step()
    assert loss_function(head(features, 1), labels, masks) < 0.05


def test_auxiliary_loss_uses_linear_soft_dice() -> None:
    """A two-class probability example distinguishes linear Dice from the squared-denominator variant."""
    probabilities = torch.tensor([[0.75, 0.25], [0.25, 0.75]])
    logits = probabilities.log()
    prediction = UniADetLogits([logits], [logits.reshape(2, 2, 1, 1)])
    labels = torch.tensor([0, 1])
    actual = UniADetLoss()(prediction, labels, labels.reshape(2, 1, 1))
    smoothed_probability = 0.750005
    expected = -math.log(0.75) - (1 - smoothed_probability) ** 2 * math.log(smoothed_probability) + 16 / 55
    assert float(actual) == pytest.approx(expected, abs=1e-6)


@pytest.mark.usefixtures("tiny_backbone")
@pytest.mark.parametrize("shots", [0, 1, 4])
def test_batch_and_memory_roundtrip(shots: int) -> None:
    """Fixed reference memory and predictions survive state-dict restoration and batching."""
    torch.manual_seed(2)
    model = UniADetModel(pretrained=False, image_size=28)
    model.head_trained.fill_(value=True)
    images = torch.rand(3, 3, 28, 28)
    if shots:
        model.set_reference_images(torch.rand(shots, 3, 28, 28))
        self_distances = model._reference_distances(model.backbone(images))  # noqa: SLF001
        assert self_distances.shape == (3, 1, 4, 4)
    batch = model(images)
    single = [model(image.unsqueeze(0)) for image in images]
    torch.testing.assert_close(batch.pred_score, torch.cat([item.pred_score for item in single]))
    torch.testing.assert_close(batch.anomaly_map, torch.cat([item.anomaly_map for item in single]))
    restored = UniADetModel(pretrained=False, image_size=28)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(images).anomaly_map, batch.anomaly_map)


@pytest.mark.usefixtures("tiny_backbone")
def test_reference_distance_and_replacement() -> None:
    """Exact references have zero distance and a replacement must discard the previous category."""
    model = UniADetModel(pretrained=False, image_size=28)
    images = torch.rand(2, 3, 28, 28)
    model.set_reference_images(images)
    distances = model._reference_distances(model.backbone(images))  # noqa: SLF001
    torch.testing.assert_close(distances, torch.zeros_like(distances), atol=2e-7, rtol=0)
    model.set_reference_images(images[:1])
    assert model.memory_bank.shape[1] == 16
    model.clear_reference_images()
    assert model.memory_bank.shape[1] == 0
    with pytest.raises(ValueError, match="At least one"):
        model.set_reference_images(images[:0])


@pytest.mark.usefixtures("tiny_backbone")
def test_legacy_fusion_has_hand_computed_nearest_distance() -> None:
    """Orthogonal unit vectors isolate nearest-neighbor reduction and both fusion weights."""
    model = UniADetModel(pretrained=False, layers=(1, 2), image_size=28, temperature=1)
    model.head_trained.fill_(value=True)
    model.backbone.grid_size = 2
    model.blur = nn.Identity()
    normal_and_anomaly = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]] * 2)
    with torch.no_grad():
        model.head.image_prototypes.copy_(normal_and_anomaly)
        model.head.pixel_prototypes.copy_(normal_and_anomaly)
    image = torch.tensor([[1.0, 0.0, 0.0]])
    patches = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]])
    features = HierarchicalFeatures([image, image], [patches, patches])
    model.memory_bank = normal_and_anomaly
    distances = model._reference_distances(features)  # noqa: SLF001
    torch.testing.assert_close(distances, torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]]))
    prediction = model.predict_from_features(features)
    probability = 1 / (1 + math.exp(-1))
    expected_corners = torch.tensor([[[[1 - probability, probability], [1 + probability, 2 - probability]]]]) / 2
    torch.testing.assert_close(prediction.anomaly_map[:, :, ::27, ::27], expected_corners)
    assert prediction.pred_score.item() == pytest.approx((1 - probability) / 2 + (1 + probability) / 4)


@pytest.mark.usefixtures("tiny_backbone")
def test_require_trained_head_and_validate_checkpoint(tmp_path: Path) -> None:
    """Random heads cannot silently produce predictions; mismatched checkpoints fail clearly."""
    model = UniADetModel(pretrained=False, image_size=28)
    with pytest.raises(RuntimeError, match="trained auxiliary head"):
        model(torch.rand(1, 3, 28, 28))
    with pytest.raises(RuntimeError, match="trained auxiliary head"):
        model.predict_from_features(model.backbone(torch.rand(1, 3, 28, 28)))
    model.head_trained.fill_(value=True)
    model.save_head(tmp_path / "head.pt")
    restored = UniADetModel(pretrained=False, image_size=28)
    restored.load_head(tmp_path / "head.pt")
    torch.testing.assert_close(restored.head.image_prototypes, model.head.image_prototypes)
    wrong = UniADetModel(pretrained=False, image_size=28, layers=(1,))
    with pytest.raises(ValueError, match="configuration"):
        wrong.load_head(tmp_path / "head.pt")


@pytest.mark.usefixtures("tiny_backbone")
@pytest.mark.parametrize("name", ["UniADet", "uniadet", "uni_adet"])
def test_model_discovery_and_learning_type(name: str) -> None:
    """Public discovery and target learning types do not change the Engine training lifecycle."""
    model = get_model(name, pretrained=False, image_size=28, visualizer=False)
    assert isinstance(model, UniADet)
    assert model.learning_type == LearningType.ZERO_SHOT
    assert UniADet(k_shot=4, pretrained=False, image_size=28).learning_type == LearningType.FEW_SHOT


@pytest.mark.parametrize("layers", [(), (2, 1), (1, 1), (0,), (25,)])
def test_invalid_layers(layers: tuple[int, ...]) -> None:
    """Invalid feature selections fail before any encoder download."""
    with pytest.raises(ValueError, match="layers"):
        UniADetBackbone(layers=layers, pretrained=False)
