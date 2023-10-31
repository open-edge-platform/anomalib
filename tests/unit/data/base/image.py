"""Unit Tests - Base Image Datamodules."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from anomalib.data import AnomalibDataModule

from .base import _TestAnomalibDataModule


class _TestAnomalibImageDatamodule(_TestAnomalibDataModule):
    # Add domain variable here.
    domain = "image"

    @pytest.mark.parametrize("subset", ["train", "val", "test"])
    def test_get_item_returns_correct_keys_and_shapes(self, datamodule: AnomalibDataModule, subset: str) -> None:
        """Test that the datamodule __getitem__ returns image, mask, label and boxes."""

        # Get the dataloader.
        dataloader = getattr(datamodule, f"{subset}_dataloader")()

        # Get the first batch.
        batch = next(iter(dataloader))

        # Check that the batch has the correct shape.
        assert batch["image"].shape == (4, 3, 256, 256)
        assert batch["label"].shape == (4,)

        # TODO: Detection task should return bounding boxes.
        # if dataloader.dataset.task == "detection":
        #     assert batch["boxes"].shape == (4, 4)

        if dataloader.dataset.task in ("detection", "segmentation"):
            assert batch["mask"].shape == (4, 256, 256)

    def test_non_overlapping_splits(self, datamodule) -> None:
        """This test ensures that all splits are non-overlapping when split mode == from_test."""
        if datamodule.val_split_mode == "from_test":
            assert (
                len(
                    set(datamodule.test_data.samples["image_path"].values).intersection(
                        set(datamodule.train_data.samples["image_path"].values)
                    )
                )
                == 0
            ), "Found train and test split contamination"
            assert (
                len(
                    set(datamodule.val_data.samples["image_path"].values).intersection(
                        set(datamodule.test_data.samples["image_path"].values)
                    )
                )
                == 0
            ), "Found train and test split contamination"

    def test_equal_splits(self, datamodule) -> None:
        """This test ensures that val and test split are equal when split mode == same_as_test."""
        if datamodule.val_split_mode == "same_as_test":
            assert all(
                datamodule.val_data.samples["image_path"].values == datamodule.test_data.samples["image_path"].values
            )
