# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - Kaputt Datamodule."""

from pathlib import Path

import polars as pl
import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import Kaputt
from anomalib.data.datasets.image.kaputt import make_kaputt_dataset
from anomalib.data.utils import LabelName, Split, ValSplitMode
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestKaputt(_TestAnomalibImageDatamodule):
    """Kaputt Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path) -> Kaputt:
        """Create and return a Kaputt datamodule."""
        datamodule_ = Kaputt(
            root=dataset_path / "kaputt",
            train_batch_size=4,
            eval_batch_size=4,
            augmentations=Resize((256, 256)),
            val_split_mode=ValSplitMode.FROM_DIR,
        )
        datamodule_.prepare_data()
        datamodule_.setup()

        return datamodule_

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "examples/configs/data/kaputt.yaml"

    @staticmethod
    def test_use_reference_adds_rows(dataset_path: Path) -> None:
        """Test that use_reference=True adds reference samples to the dataset."""
        root = dataset_path / "kaputt"
        samples_without = make_kaputt_dataset(root, split=Split.TRAIN, use_reference=False)
        samples_with = make_kaputt_dataset(root, split=Split.TRAIN, use_reference=True)

        assert len(samples_with) > len(samples_without)
        # All reference rows should be normal
        added = samples_with.filter(~pl.col("image_path").is_in(samples_without["image_path"]))
        assert (added["label"] == "normal").all()
        assert (added["label_index"] == int(LabelName.NORMAL)).all()
        # Reference paths must point into reference-image/
        assert added["image_path"].str.contains("reference-image").all()

    @staticmethod
    def test_image_type_crop_switches_subdirectory(dataset_path: Path) -> None:
        """Test that image_type='crop' reads from query-crop/ instead of query-image/."""
        root = dataset_path / "kaputt"
        samples_image = make_kaputt_dataset(root, split=Split.TRAIN, image_type="image")
        samples_crop = make_kaputt_dataset(root, split=Split.TRAIN, image_type="crop")

        assert len(samples_image) == len(samples_crop)
        assert samples_image["image_path"].str.contains("query-image").all()
        assert samples_crop["image_path"].str.contains("query-crop").all()

    @staticmethod
    def test_abnormal_samples_have_mask_path(dataset_path: Path) -> None:
        """Test that abnormal samples have non-empty mask_path and normals do not."""
        root = dataset_path / "kaputt"
        # Use val split which has both normal and abnormal in the dummy data
        samples = make_kaputt_dataset(root, split=Split.VAL)

        abnormal = samples.filter(pl.col("label_index") == int(LabelName.ABNORMAL))
        normal = samples.filter(pl.col("label_index") == int(LabelName.NORMAL))

        assert len(abnormal) > 0, "Expected abnormal samples in val split"
        assert len(normal) > 0, "Expected normal samples in val split"
        assert (abnormal["mask_path"] != "").all(), "All abnormal samples should have a mask_path"
        assert (normal["mask_path"] == "").all(), "No normal sample should have a mask_path"

    @staticmethod
    def test_use_reference_crop_paths(dataset_path: Path) -> None:
        """Test that use_reference=True with image_type='crop' uses reference-crop/."""
        root = dataset_path / "kaputt"
        samples = make_kaputt_dataset(root, split=Split.TRAIN, image_type="crop", use_reference=True)

        ref_rows = samples.filter(pl.col("image_path").str.contains("reference-"))
        assert len(ref_rows) > 0
        assert ref_rows["image_path"].str.contains("reference-crop").all()
