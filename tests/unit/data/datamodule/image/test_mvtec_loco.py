# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - MVTecLoco Datamodule."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from torchvision.transforms.v2 import Resize

from anomalib.data import MVTecLOCO, MVTecLOCODataset
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestMVTecLOCO(_TestAnomalibImageDatamodule):
    """MVTecLOCO Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, make_dummy_dataset: Callable[[str], Path]) -> MVTecLOCO:
        """Create and return a MVTecLOCO datamodule."""
        make_dummy_dataset("mvtec_loco")
        datamodule = MVTecLOCO(
            root=dataset_path / "mvtec_loco",
            category="dummy",
            train_batch_size=4,
            eval_batch_size=4,
            augmentations=Resize((256, 256)),
        )
        datamodule.prepare_data()
        datamodule.setup()

        return datamodule

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "examples/configs/data/mvtec_loco.yaml"

    @staticmethod
    def test_mask_is_binary(datamodule: MVTecLOCO) -> None:
        """Test if the mask tensor is binary."""
        mask_tensor = datamodule.test_data[0].gt_mask
        if mask_tensor is not None:
            is_binary = (mask_tensor.eq(0) | mask_tensor.eq(1)).all()
            assert is_binary.item() is True


def test_multiple_masks_are_associated_and_combined(tmp_path: Path) -> None:
    """Anomalous images keep all their mask paths and the masks are combined; normal images keep no mask."""
    root = tmp_path / "dummy"
    size = 16
    image = np.zeros((size, size, 3), dtype=np.uint8)
    for image_path in (
        root / "train/good/000.png",
        root / "test/good/000.png",
        root / "test/logical_anomalies/000.png",
    ):
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(image_path)

    mask_dir = root / "ground_truth/logical_anomalies/000"
    mask_dir.mkdir(parents=True)
    first_mask = np.zeros((size, size), dtype=np.uint8)
    first_mask[:4, :4] = 255
    second_mask = np.zeros((size, size), dtype=np.uint8)
    second_mask[-4:, -4:] = 255
    Image.fromarray(first_mask).save(mask_dir / "000.png")
    Image.fromarray(second_mask).save(mask_dir / "001.png")

    dataset = MVTecLOCODataset(root=tmp_path, category="dummy", split="test")
    samples = dataset.samples.set_index("label")

    assert sorted(samples.loc["logical_anomalies", "mask_path"]) == [
        str(mask_dir / "000.png"),
        str(mask_dir / "001.png"),
    ]
    assert samples.loc["good", "mask_path"] == ""

    anomalous_index = dataset.samples.index[dataset.samples.label == "logical_anomalies"][0]
    gt_mask = dataset[anomalous_index].gt_mask
    assert gt_mask.shape == (size, size)
    assert gt_mask[:4, :4].all()
    assert gt_mask[-4:, -4:].all()
    assert gt_mask.sum() == 2 * 4 * 4
