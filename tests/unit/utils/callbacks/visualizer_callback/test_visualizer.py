"""Test visualizer callback."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.loggers import AnomalibTensorBoardLogger
from anomalib.utils.types import TaskType
from anomalib.utils.visualization.image import ImageVisualizer

from .dummy_lightning_model import DummyModule


@pytest.mark.parametrize("task", [TaskType.CLASSIFICATION, TaskType.SEGMENTATION, TaskType.DETECTION])
def test_add_images(task: TaskType, dataset_path: Path) -> None:
    """Tests if tensorboard logs are generated."""
    with tempfile.TemporaryDirectory() as dir_loc:
        logger = AnomalibTensorBoardLogger(name="tensorboard_logs", save_dir=dir_loc)
        model = DummyModule(dataset_path)
        engine = Engine(
            logger=logger,
            enable_checkpointing=False,
            default_root_dir=dir_loc,
            task=task,
            limit_test_batches=1,
            log_image=True,
            visualization_generators=ImageVisualizer(),
        )
        engine.test(model=model, datamodule=MVTec(root=dataset_path / "mvtec", category="dummy"))
        # test if images are logged
        assert len(list(Path(dir_loc).glob("**/*.png"))) == 1, "Failed to save to local path"

        # test if tensorboard logs are created
        assert len(list((Path(dir_loc) / "tensorboard_logs").glob("version_*"))) != 0, "Failed to save to tensorboard"
