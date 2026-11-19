# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline regressions for author-head conversion and source-aligned evaluation."""

import csv
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image
from tools.uniadet.author import convert_head
from tools.uniadet.diagnose import scoring_variants
from tools.uniadet.reproduce import (
    EvaluationDataset,
    Sample,
    binary_metrics,
    category_samples,
    reference_indices,
    use_author_clip_precision,
)

from anomalib.models.image.uniadet.backbones import HierarchicalFeatures, UniADetBackbone
from anomalib.models.image.uniadet.head import MSDCSHead
from anomalib.models.image.uniadet.torch_model import UniADetModel


@pytest.mark.usefixtures("tiny_backbone")
@pytest.mark.parametrize("shots", [0, 4])
def test_author_scoring_matches_prespecified_probe(shots: int) -> None:
    """Production author scoring must agree with the independently tested diagnostic combination."""
    torch.manual_seed(10)
    model = UniADetModel(pretrained=False, image_size=28, protocol="author")
    model.head_trained.fill_(value=True)
    if shots:
        model.set_reference_images(torch.rand(shots, 3, 28, 28))
    features = model.backbone(torch.rand(3, 3, 28, 28))
    expected = scoring_variants(model, features)["published_executable"]
    actual = model.predict_from_features(features)
    torch.testing.assert_close(actual.pred_score, expected.pred_score)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map)


@pytest.mark.usefixtures("tiny_backbone")
def test_head_protocol_cannot_be_silently_swapped(tmp_path: Path) -> None:
    """Legacy normalized-feature heads and author raw-feature heads are not interchangeable."""
    legacy = UniADetModel(pretrained=False, image_size=28)
    author = UniADetModel(pretrained=False, image_size=28, protocol="author")
    for model, other in ((legacy, author), (author, legacy)):
        model.save_head(tmp_path / "head.pt")
        with pytest.raises(ValueError, match="configuration"):
            other.load_head(tmp_path / "head.pt")
        model.load_head(tmp_path / "head.pt")


def test_author_head_conversion_preserves_values_and_rejects_malformed_tensors() -> None:
    """Import stacks tensors without reordering classes, changing values or accepting extra parameters."""
    state = {
        f"{kind}_prompts.{index}.weight": torch.arange(6).reshape(2, 3).float() + index + 1
        for kind in ("class", "patch")
        for index in range(5)
    }
    converted = convert_head({"msdcs": state}, feature_dim=3)
    torch.testing.assert_close(converted["image_prototypes"][2], state["class_prompts.2.weight"])
    torch.testing.assert_close(converted["pixel_prototypes"][4], state["patch_prompts.4.weight"])
    state["class_prompts.0.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        convert_head({"msdcs": state}, feature_dim=3)
    with pytest.raises(ValueError, match="exactly"):
        convert_head({"msdcs": {}}, feature_dim=3)


def test_reference_sampler_matches_author_global_rng_without_advancing_it() -> None:
    """CPU randint is reset independently per category, retains duplicate draws and leaves callers alone."""
    torch.manual_seed(20)
    expected = torch.randint(0, 3, (4,)).tolist()
    before = torch.random.get_rng_state()
    assert reference_indices(3, 4, 20) == expected
    assert reference_indices(1, 4, 10) == [0, 0, 0, 0]
    assert torch.equal(before, torch.random.get_rng_state())


def test_visa_original_csv_order_and_confinement(tmp_path: Path) -> None:
    """Reference indices address CSV order, not lexicographic prepared filenames, and cannot escape root."""
    root = tmp_path / "dataset"
    (root / "split_csv").mkdir(parents=True)
    for name in ("b.png", "a.png"):
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(root / name)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / "outside.png")
    csv_path = root / "split_csv/1cls.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["object", "split", "label", "image", "mask"])
        writer.writeheader()
        for name in ("b.png", "a.png"):
            writer.writerow({"object": "candle", "split": "train", "label": "normal", "image": name, "mask": ""})
    samples = category_samples(root, "visa", "candle", "train")
    assert [sample.image.name for sample in samples] == ["b.png", "a.png"]
    csv_path.write_text(csv_path.read_text().replace("b.png", "../outside.png"))
    with pytest.raises(ValueError, match=r"within|outside|traversal"):
        category_samples(root, "visa", "candle", "train")


def test_original_mask_is_binarized_before_nearest_resize(tmp_path: Path) -> None:
    """VisA instance-valued masks become binary before the two-stage evaluation resize."""
    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image)
    pixels = np.zeros((8, 8), dtype=np.uint8)
    pixels[2:6, 2:6] = 3
    Image.fromarray(pixels).save(mask)
    sample = EvaluationDataset([Sample(image, mask, 1)], 16)[0]
    assert sample.gt_mask.sum() == 64
    assert sample.gt_mask.dtype == torch.bool


def test_clip_author_positions_keep_native_grid_and_shifted_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct timm resizing uses a different antialias/coordinate convention from the detection checkpoint."""
    encoder = MagicMock()
    encoder.patch_embed.grid_size = (24, 24)
    original = torch.arange(577 * 2).reshape(1, 577, 2).float()
    encoder.pos_embed = torch.nn.Parameter(original.clone())
    factory = MagicMock(return_value=encoder)
    monkeypatch.setattr("anomalib.models.image.uniadet.backbones.timm.create_model", factory)
    UniADetBackbone("clip", pretrained=False, protocol="author")
    assert factory.call_args.kwargs["img_size"] == 336
    encoder.patch_embed.set_input_size.assert_called_once_with(img_size=518)
    assert encoder.pos_embed.shape == (1, 1370, 2)
    torch.testing.assert_close(encoder.pos_embed[:, :1], original[:, :1])
    ordinary = (
        torch.nn.functional.interpolate(
            original[:, 1:].reshape(1, 24, 24, 2).permute(0, 3, 1, 2),
            (37, 37),
            mode="bicubic",
            align_corners=False,
        )
        .flatten(2)
        .transpose(1, 2)
    )
    assert not torch.allclose(encoder.pos_embed[:, 1:], ordinary)


def test_mixed_precision_features_use_fp32_prototype_math() -> None:
    """The author CLIP encoder emits FP16 while its head explicitly computes logits in FP32."""
    head = MSDCSHead(3, 1)
    image = torch.tensor([[1.0, 0.5, 0.25]], dtype=torch.float16)
    actual = head(HierarchicalFeatures([image], [image[:, None]]), 1)
    expected = head(HierarchicalFeatures([image.float()], [image[:, None].float()]), 1)
    torch.testing.assert_close(actual.image[0], expected.image[0])
    torch.testing.assert_close(actual.pixel[0], expected.pixel[0])


def test_author_clip_layernorm_keeps_unrounded_parameters_and_accepts_half() -> None:
    """Restoring FP32 weights alone is insufficient: LayerNorm must explicitly upcast its half input."""
    encoder = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.LayerNorm(3)).eval()
    with torch.no_grad():
        encoder[1].weight.copy_(torch.tensor([1.00013, 0.91234, 1.25678]))
    original = encoder[1].weight.detach().clone()
    use_author_clip_precision(encoder)
    torch.testing.assert_close(encoder[1].weight, original, rtol=0, atol=0)
    assert encoder[0].weight.dtype == torch.float16
    result = encoder(torch.randn(2, 3).half())
    assert result.dtype == torch.float16
    assert torch.isfinite(result).all()


@pytest.mark.parametrize(
    ("target", "scores", "auroc", "paper_aupr", "average_precision"),
    [
        ([1, 0, 1], [0.9, 0.8, 0.7], 0.5, 19 / 24, 5 / 6),
        ([0, 1], [0.5, 0.5], 0.5, 0.75, 0.5),
    ],
)
def test_paper_pr_area_is_not_standard_average_precision(
    target: list[int],
    scores: list[float],
    auroc: float,
    paper_aupr: float,
    average_precision: float,
) -> None:
    """Hand-computed ranking/tie cases keep the author's trapezoid and standard AP visibly distinct."""
    actual = binary_metrics(torch.tensor(scores), torch.tensor(target))
    assert actual["AUROC"] == pytest.approx(auroc)
    assert actual["AUPR"] == pytest.approx(paper_aupr)
    assert actual["average_precision"] == pytest.approx(average_precision)
