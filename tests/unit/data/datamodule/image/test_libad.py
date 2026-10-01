# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - LIBAD Datamodule."""

from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import LIBAD
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestLIBAD(_TestAnomalibImageDatamodule):
    """LIBAD Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path) -> LIBAD:
        """Create and return a LIBAD datamodule."""
        datamodule_ = LIBAD(
            root=dataset_path / "libad",
            category="1_wrinkling",
            modality="A",
            train_batch_size=4,
            eval_batch_size=4,
            augmentations=Resize((256, 256)),
            test_split_ratio=0.2,
        )

        datamodule_.prepare_data()
        datamodule_.setup()

        return datamodule_

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "examples/configs/data/libad.yaml"
