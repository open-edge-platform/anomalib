"""Datamodule Unit Tests."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from anomalib.data import AnomalibDataModule
import pytest
from torch.utils.data import DataLoader


class _TestAnomalibDataModule:
    """Base test class for ``AnomalibDataModule``.

    This is a base class for testing the AnomalibDataModule. Since
    ``AnomalibDataModule`` has methods that are yet to be implemented, this base
    test class is not meant to be used directly.
    """

    @pytest.mark.parametrize("subset", ["train", "val", "test"])
    def test_datamodule_has_dataloader_attributes(self, datamodule: AnomalibDataModule, subset: str) -> None:
        """Test that the datamodule has the correct dataloader attributes."""
        dataloader = f"{subset}_dataloader"
        assert hasattr(datamodule, dataloader) and isinstance(getattr(datamodule, dataloader)(), DataLoader)
