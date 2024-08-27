"""Unit Tests - Base Depth Datamodules."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields

import pytest

from anomalib.data import AnomalibDataModule

from .base import _TestAnomalibDataModule


class _TestAnomalibDepthDatamodule(_TestAnomalibDataModule):
    @pytest.mark.parametrize("subset", ["train", "val", "test"])
    def test_get_item_returns_correct_keys_and_shapes(self, datamodule: AnomalibDataModule, subset: str) -> None:
        """Test that the datamodule __getitem__ returns the correct keys and shapes."""
        # Get the dataloader.
        dataloader = getattr(datamodule, f"{subset}_dataloader")()

        # Get the first batch.
        batch = next(iter(dataloader))

        # Check that the batch has the correct keys.
        expected_fields = {"image_path", "depth_path", "gt_label", "image", "depth_map"}

        if dataloader.dataset.task in ("detection", "segmentation"):
            expected_fields |= {"mask_path", "gt_mask"}

            if dataloader.dataset.task == "detection":
                expected_fields |= {"boxes"}

        batch_fields = {field.name for field in fields(batch) if getattr(batch, field.name) is not None}
        assert batch_fields == expected_fields

        # Check that the batch has the correct shape.
        assert len(batch.image_path) == 4
        assert len(batch.depth_path) == 4
        assert batch.image.shape == (4, 3, 256, 256)
        assert batch.depth_map.shape == (4, 3, 256, 256)
        assert batch.gt_label.shape == (4,)

        if dataloader.dataset.task in ("detection", "segmentation"):
            assert batch.gt_mask.shape == (4, 256, 256)
