# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - Tabular Datamodule."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import Folder, Tabular
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestTabular(_TestAnomalibImageDatamodule):
    """Tabular Datamodule Unit Tests."""

    @staticmethod
    def get_samples_dataframe(dataset_path: Path) -> pd.DataFrame:
        """Create samples DataFrame using the Folder datamodule."""
        datamodule_ = Folder(
            name="dummy",
            root=dataset_path / "mvtecad" / "dummy",
            normal_dir="train/good",
            abnormal_dir="test/bad",
            normal_test_dir="test/good",
            mask_dir="ground_truth/bad",
            train_batch_size=4,
            eval_batch_size=4,
            num_workers=0,
        )
        datamodule_.setup()
        return pd.concat([
            datamodule_.train_data.samples,
            datamodule_.test_data.samples,
            datamodule_.val_data.samples,
        ])

    @pytest.fixture(
        params=[
            None,
            ["label"],
            ["label_index"],
            ["split"],
            ["mask_path"],
        ],
    )
    @staticmethod
    def columns_to_drop(request: pytest.FixtureRequest) -> list[str] | None:
        """Return the columns to be dropped from the samples dataframe."""
        return request.param

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path, columns_to_drop: list | None) -> Tabular:
        """Create and return a Tabular datamodule."""
        samples = TestTabular.get_samples_dataframe(dataset_path)
        if columns_to_drop:
            samples = samples.drop(columns_to_drop, axis="columns")
        datamodule_ = Tabular(
            name="dummy",
            samples=samples,
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
        return "examples/configs/data/tabular.yaml"


class TestTabularFromFile(TestTabular):
    """Tabular Datamodule Unit Tests for alternative constructor.

    Tests for the Datamodule creation from file.
    """

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path) -> Tabular:
        """Create and return a Tabular datamodule."""
        samples = TestTabular.get_samples_dataframe(dataset_path)
        with tempfile.NamedTemporaryFile(suffix=".csv") as samples_file:
            samples.to_csv(samples_file)
            samples_file.seek(0)

            datamodule_ = Tabular.from_file(
                name="dummy",
                file_path=samples_file.name,
                train_batch_size=4,
                eval_batch_size=4,
                num_workers=0,
                augmentations=Resize((256, 256)),
            )
            datamodule_.setup()

        return datamodule_
