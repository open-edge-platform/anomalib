# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit Tests - Kaputt Datamodule."""

from pathlib import Path

import pytest
from torchvision.transforms.v2 import Resize

from anomalib.data import Kaputt
from anomalib.data.utils import ValSplitMode
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
