# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - BMAD Datamodule."""

from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import BMAD
from anomalib.data.utils import ValSplitMode
from tests.unit.data.datamodule.base.image import _TestAnomalibImageDatamodule


class TestBMAD(_TestAnomalibImageDatamodule):
    """BMAD Datamodule Unit Tests."""

    @pytest.fixture()
    @staticmethod
    def datamodule(dataset_path: Path) -> BMAD:
        """Create and return a BMAD datamodule."""
        datamodule_ = BMAD(
            root=dataset_path / "bmad",
            category="dummy",
            train_batch_size=4,
            eval_batch_size=4,
            val_split_mode=ValSplitMode.FROM_DIR,
            augmentations=Resize((256, 256)),
        )
        datamodule_.prepare_data()
        datamodule_.setup()

        return datamodule_

    @pytest.fixture()
    @staticmethod
    def fxt_data_config_path() -> str:
        """Return the path to the test data config."""
        return "examples/configs/data/bmad.yaml"
