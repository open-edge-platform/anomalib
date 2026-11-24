# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for auxiliary supervision and normal-only reference selection."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from tools.uniadet.data import AuxiliaryDataset, CalibrationData
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Resize

from anomalib.data import ImageItem
from anomalib.models import UniADet


class LabeledImages(Dataset):
    """Small deterministic source with one normal and one anomalous image."""

    def __init__(self) -> None:
        self.samples = pd.DataFrame({"label_index": [0, 1]})

    def __len__(self) -> int:
        """Return the sample count."""
        return 2

    def __getitem__(self, index: int) -> ImageItem:
        """Return a fresh item so in-place augmentation cannot leak across samples."""
        mask = torch.zeros(30, 30)
        if index:
            mask[3:23, 5:25] = 1
        return ImageItem(image=mask.expand(3, -1, -1).clone(), gt_mask=mask, gt_label=index)


@pytest.mark.parametrize("grid", [2, 3])
@pytest.mark.parametrize("label", [0, 1])
def test_class_aware_augmentation_preserves_supervision(grid: int, label: int) -> None:
    """Mosaic and crop retain label/mask consistency and cannot contaminate normal examples."""
    torch.manual_seed(7)
    dataset = AuxiliaryDataset.__new__(AuxiliaryDataset)
    dataset.resize = Resize((30, 30), antialias=False)
    dataset.datasets = [LabeledImages()]
    dataset.normal_indices = [[0]]
    item = dataset.datasets[0][label]
    for result in (dataset._mosaic(item, 0, grid), dataset._crop(item, grid)):  # noqa: SLF001
        assert result.image.shape == (3, 30, 30)
        assert result.gt_mask.shape == (30, 30)
        assert result.gt_label == label
        assert result.gt_mask.any() == bool(label)
        if not label:
            assert not result.image.any()


def test_calibration_splits_are_disjoint() -> None:
    """Quantizer fitting and accuracy control never consume the same examples."""
    data = CalibrationData(LabeledImages(), subset_size=1)
    assert set(data.calibration.indices).isdisjoint(data.validation.indices)
    with pytest.raises(ValueError, match="disjoint"):
        CalibrationData(LabeledImages(), subset_size=2)


def test_training_transforms_preserve_masks_and_leave_calibration_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Photometric and geometric training transforms run even when class-aware augmentation is skipped."""
    monkeypatch.setattr("tools.uniadet.data.MVTecADDataset", lambda **_kwargs: LabeledImages())
    dataset = AuxiliaryDataset(tmp_path, "mvtec", 30)
    monkeypatch.setattr(dataset, "_mosaic", lambda item, _category, _grid: item)
    monkeypatch.setattr(dataset, "_crop", lambda item, _grid: item)
    source = dataset.datasets[0][1]
    changed_photometry = False
    changed_geometry = False
    for seed in range(12):
        torch.manual_seed(seed)
        item = dataset[1]
        intensity = item.image.mean(dim=0)
        foreground = intensity > (intensity.min() + intensity.max()) / 2
        assert torch.equal(foreground, item.gt_mask.bool())
        assert item.gt_label == source.gt_label
        changed_photometry |= not torch.all((item.image == 0) | (item.image == 1))
        changed_geometry |= not torch.equal(item.gt_mask, source.gt_mask)
    assert changed_photometry
    assert changed_geometry
    dataset.augmentation = False
    torch.testing.assert_close(dataset[1].image, source.image)
    torch.testing.assert_close(dataset[1].gt_mask, source.gt_mask)


@pytest.mark.usefixtures("tiny_backbone")
def test_reference_directory_setup_and_restore(tmp_path: Path) -> None:
    """References initialize once and a restored memory bank needs no source files."""
    for index in range(3):
        Image.fromarray(np.full((28, 28, 3), 40 + index * 30, dtype=np.uint8)).save(tmp_path / f"{index}.png")
    model = UniADet(pretrained=False, image_size=28, k_shot=2, few_shot_source=tmp_path)
    model.on_predict_start()
    assert model.model.memory_bank.shape == (5, 32, 3)
    restored = UniADet(pretrained=False, image_size=28, k_shot=2)
    restored.load_state_dict(model.state_dict())
    restored.on_predict_start()
    torch.testing.assert_close(restored.model.memory_bank, model.model.memory_bank)
    insufficient = UniADet(pretrained=False, image_size=28, k_shot=4, few_shot_source=tmp_path)
    with pytest.raises(ValueError, match="found only 3 normal"):
        insufficient.on_predict_start()


@pytest.mark.usefixtures("tiny_backbone")
def test_reference_directory_rejects_symlink_escape(tmp_path: Path) -> None:
    """Untrusted reference-directory contents cannot read images outside the selected root."""
    outside = tmp_path / "outside.png"
    Image.fromarray(np.zeros((28, 28, 3), dtype=np.uint8)).save(outside)
    references = tmp_path / "references"
    references.mkdir()
    (references / "escape.png").symlink_to(outside)
    model = UniADet(pretrained=False, image_size=28, k_shot=1, few_shot_source=references)
    with pytest.raises(ValueError, match="outside the allowed directory"):
        model.on_predict_start()
