# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - Folder Datamodule."""

from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import Folder
from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.utils import ValSplitMode
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestFolder(_TestAnomalibImageDatamodule):
    """Folder Datamodule Unit Tests.

    All of the folder datamodule tests are placed in ``TestFolder`` class.
    """

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path) -> Folder:
        """Create and return a Folder datamodule."""
        # expects a relative directory to the root.
        mask_dir = "ground_truth/bad"

        # Create and prepare the dataset
        datamodule_ = Folder(
            name="dummy",
            root=dataset_path / "mvtecad" / "dummy",
            normal_dir="train/good",
            abnormal_dir="test/bad",
            normal_test_dir="test/good",
            mask_dir=mask_dir,
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            augmentations=Resize((256, 256)),
        )
        datamodule_.setup()

        return datamodule_

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "examples/configs/data/folder.yaml"

    @staticmethod
    def test_synthetic_blend_factor_is_forwarded(dataset_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure datamodule forwards synthetic blend factor into synthetic split creation."""
        captured: dict[str, float | tuple[float, float]] = {}

        def _spy_from_dataset(
            cls: type[object],
            dataset: AnomalibDataset,
            blend_factor: float | tuple[float, float] = (0.01, 0.2),
        ) -> AnomalibDataset:
            del cls
            captured["blend_factor"] = blend_factor
            # Return original dataset to keep setup flow lightweight.
            return dataset

        monkeypatch.setattr(
            "anomalib.data.datamodules.base.image.SyntheticAnomalyDataset.from_dataset",
            classmethod(_spy_from_dataset),
        )

        datamodule = Folder(
            name="dummy",
            root=dataset_path / "mvtecad" / "dummy",
            normal_dir="train/good",
            abnormal_dir="test/bad",
            normal_test_dir="test/good",
            mask_dir="ground_truth/bad",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            augmentations=Resize((256, 256)),
            val_split_mode=ValSplitMode.SYNTHETIC,
            synthetic_blend_factor=(0.3, 0.4),
        )
        datamodule.setup()

        assert captured["blend_factor"] == (0.3, 0.4)

    @staticmethod
    def test_synthetic_blend_factor_is_forwarded_for_test_split(
        dataset_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ensure test synthetic split also receives datamodule blend factor."""
        captured: list[float | tuple[float, float]] = []

        def _spy_from_dataset(
            cls: type[object],
            dataset: AnomalibDataset,
            blend_factor: float | tuple[float, float] = (0.01, 0.2),
        ) -> AnomalibDataset:
            del cls
            captured.append(blend_factor)
            return dataset

        monkeypatch.setattr(
            "anomalib.data.datamodules.base.image.SyntheticAnomalyDataset.from_dataset",
            classmethod(_spy_from_dataset),
        )

        datamodule = Folder(
            name="dummy",
            root=dataset_path / "mvtecad" / "dummy",
            normal_dir="train/good",
            abnormal_dir="test/bad",
            normal_test_dir="test/good",
            mask_dir="ground_truth/bad",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
            augmentations=Resize((256, 256)),
            test_split_mode="synthetic",
            val_split_mode=ValSplitMode.NONE,
            synthetic_blend_factor=0.25,
        )
        datamodule.setup()

        assert captured == [0.25]
