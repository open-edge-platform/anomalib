"""Unit Tests - Avenue Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from anomalib.data import Avenue, TaskType

from .base import _TestAnomalibVideoDatamodule


class TestAvenue(_TestAnomalibVideoDatamodule):
    """Avenue Datamodule Unit Tests."""

    @pytest.fixture
    def datamodule(self, task_type: TaskType) -> Avenue:
        # Create and prepare the dataset

        _datamodule = Avenue(
            root="./datasets/avenue",
            gt_dir="./datasets/avenue/ground_truth_demo",
            image_size=256,
            task=task_type,
            num_workers=0,
            train_batch_size=4,
            eval_batch_size=4,
        )

        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule
