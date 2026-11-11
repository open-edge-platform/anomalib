# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for MAD-Sim splits, exact mask matching, and public datamodule API."""

from collections.abc import Callable
from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import MADSim, MADSimDataset
from anomalib.data.datasets.image.mad_sim import CATEGORIES, make_mad_sim_dataset
from anomalib.data.utils import ValSplitMode
from tests.helpers.data import DummyImageDatasetGenerator
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestMADSim(_TestAnomalibImageDatamodule):
    """Exercise dataloaders and config discovery using the shared dummy fixture."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, make_dummy_dataset: Callable[[str], Path]) -> MADSim:
        """Create a datamodule using the official directory layout."""
        make_dummy_dataset("mad_sim")
        datamodule = MADSim(
            root=dataset_path / "mad_sim",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            augmentations=Resize((256, 256)),
        )
        datamodule.prepare_data()
        datamodule.setup()
        return datamodule

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the public MAD-Sim config."""
        return "examples/configs/data/mad_sim.yaml"

    @staticmethod
    def test_official_splits(datamodule: MADSim) -> None:
        """Default setup keeps every official training/test image in its split."""
        for split, dataset in (("train", datamodule.train_data), ("test", datamodule.test_data)):
            expected = {str(path.resolve()) for path in (datamodule.root / "01Gorilla" / split).glob("*/*.png")}
            assert set(dataset.samples.image_path) == expected
            assert dataset.task == "segmentation"
        assert (datamodule.train_data.samples.label_index == 0).all()
        assert set(datamodule.test_data.samples.label_index) == {0, 1}
        assert set(datamodule.train_data.samples.image_path).isdisjoint(datamodule.test_data.samples.image_path)
        assert not datamodule.train_data[0].gt_mask.any()
        abnormal = datamodule.test_data.samples.query("label_index == 1").index[0]
        assert datamodule.test_data[abnormal].gt_mask.any()

    @staticmethod
    def test_validation_from_test(datamodule: MADSim) -> None:
        """Optional validation splitting reuses base logic without contamination."""
        original_test = set(datamodule.test_data.samples.image_path)
        split_module = MADSim(root=datamodule.root, val_split_mode=ValSplitMode.FROM_TEST, seed=42)
        split_module.setup()
        test = set(split_module.test_data.samples.image_path)
        val = set(split_module.val_data.samples.image_path)
        assert test.isdisjoint(val)
        assert test | val == original_test

    @staticmethod
    def test_no_test_split_preserves_all_samples(datamodule: MADSim) -> None:
        """Disabling test splitting preserves normal and anomalous test samples."""
        original_train = set(datamodule.train_data.samples.image_path)
        original_test = set(datamodule.test_data.samples.image_path)
        unsplit = MADSim(root=datamodule.root, test_split_mode="none", num_workers=0)
        unsplit.setup()
        assert set(unsplit.train_data.samples.image_path) == original_train
        assert set(unsplit.test_data.samples.image_path) == original_test
        assert set(unsplit.test_data.samples.label_index) == {0, 1}


@pytest.fixture()
def category_root(tmp_path: Path) -> Path:
    """Generate enough unpadded filenames to expose lexicographic mask mismatches."""
    DummyImageDatasetGenerator(
        data_format="mad_sim",
        root=tmp_path,
        num_train=2,
        num_test=12,
        image_shape=(32, 32),
    ).generate_dataset()
    return tmp_path / "mad_sim" / "01Gorilla"


def test_exact_mask_pairs(category_root: Path) -> None:
    """Repeated stems in different defect folders must match their own masks."""
    samples = make_mad_sim_dataset(category_root)
    assert len(samples) == 50
    for sample in samples.itertuples():
        if sample.label_index == 0:
            assert sample.mask_path == ""
        else:
            image = Path(sample.image_path)
            expected = category_root / "ground_truth" / image.parent.name / f"{image.stem}_mask.png"
            assert Path(sample.mask_path) == expected


@pytest.mark.parametrize("category", CATEGORIES)
def test_categories(category_root: Path, category: str) -> None:
    """All 20 official category names work in both public APIs."""
    renamed = category_root.with_name(category)
    if renamed != category_root:
        category_root.rename(renamed)
    datamodule = MADSim(root=renamed.parent, category=category)
    datamodule.prepare_data()
    assert len(MADSimDataset(root=renamed.parent, category=category, split="train")) == 2


@pytest.mark.parametrize("category", ["gorilla", "../outside", ""])
@pytest.mark.parametrize("constructor", [MADSim, MADSimDataset])
def test_invalid_category(category: str, constructor: type) -> None:
    """Reject typos and traversal before accessing the filesystem."""
    with pytest.raises(ValueError, match="Available categories"):
        constructor(category=category)


def test_missing_mask(category_root: Path) -> None:
    """A missing mask fails at setup, even if another mask has a similar stem."""
    (category_root / "ground_truth" / "Burrs" / "1_mask.png").unlink()
    with pytest.raises(FileNotFoundError, match=r"1_mask\.png"):
        make_mad_sim_dataset(category_root, "test")
    assert len(make_mad_sim_dataset(category_root, "train")) == 2


def test_invalid_split(category_root: Path) -> None:
    """Reject unsupported split names instead of silently returning no samples."""
    with pytest.raises(ValueError, match="Unknown MAD-Sim split"):
        make_mad_sim_dataset(category_root, "val")


def test_missing_data(tmp_path: Path) -> None:
    """Missing data gives actionable download instructions."""
    with pytest.raises(FileNotFoundError, match=r"Download MAD-Sim\.zip"):
        MADSim(root=tmp_path).prepare_data()


def test_no_validation_directory() -> None:
    """MAD-Sim does not advertise an official validation split."""
    with pytest.raises(ValueError, match="no official validation directory"):
        MADSim(val_split_mode="from_dir")


def test_empty_split(tmp_path: Path) -> None:
    """Empty split directories cannot silently create unusable datasets."""
    (tmp_path / "train").mkdir()
    with pytest.raises(RuntimeError, match="Found 0 images"):
        make_mad_sim_dataset(tmp_path, "train")


@pytest.mark.parametrize("relative_path", ["test/Burrs/0.png", "ground_truth/Burrs/0_mask.png"])
def test_symlink_escape(category_root: Path, tmp_path: Path, relative_path: str) -> None:
    """Reject images and masks resolving outside the category directory."""
    path = category_root / relative_path
    outside = tmp_path / "outside.png"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks requires OS privileges")
    with pytest.raises(ValueError, match="outside the allowed directory"):
        make_mad_sim_dataset(category_root, "test")
