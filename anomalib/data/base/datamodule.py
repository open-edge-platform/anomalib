"""Anomalib datamodule base class."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from abc import ABC
from typing import Optional

from pandas import DataFrame
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader

from anomalib.data.base.dataset import AnomalibDataset
from anomalib.data.synthetic import SyntheticValidationSet
from anomalib.data.utils import ValSplitMode, random_split

logger = logging.getLogger(__name__)


class AnomalibDataModule(LightningDataModule, ABC):
    """Base Anomalib data module.

    Args:
        train_batch_size (int): Batch size used by the train dataloader.
        test_batch_size (int): Batch size used by the val and test dataloaders.
        num_workers (int): Number of workers used by the train, val and test dataloaders.
    """

    def __init__(self, train_batch_size: int, eval_batch_size: int, num_workers: int, val_split_mode: ValSplitMode):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.val_split_mode = val_split_mode

        self.train_data: Optional[AnomalibDataset] = None
        self.val_data: Optional[AnomalibDataset] = None
        self.test_data: Optional[AnomalibDataset] = None

        self._samples: Optional[DataFrame] = None

    def setup(self, stage: Optional[str] = None):
        """Setup train, validation and test data.

        Args:
          stage: Optional[str]:  Train/Val/Test stages. (Default value = None)
        """
        if not self.is_setup:
            self._setup(stage)
        assert self.is_setup

    def _setup(self, _stage: Optional[str] = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        May be overridden in subclass for custom splitting behaviour.
        """
        assert self.train_data is not None
        assert self.test_data is not None

        self.train_data.setup()
        self.test_data.setup()
        if self.val_split_mode == ValSplitMode.FROM_TEST:
            self.val_data, self.test_data = random_split(self.test_data, [0.5, 0.5], label_aware=True)
        elif self.val_split_mode == ValSplitMode.SAME_AS_TEST:
            self.val_data = self.test_data
        elif self.val_split_mode == ValSplitMode.SYNTHETIC:
            self.train_data, normal_val_data = random_split(self.train_data, 0.5)
            self.val_data = SyntheticValidationSet.from_dataset(normal_val_data)
        else:
            raise ValueError(f"Unknown validation split mode: {self.val_split_mode}")

    @property
    def is_setup(self):
        """Checks if setup() has been called."""
        if self.train_data is None or self.val_data is None or self.test_data is None:
            return False
        return self.train_data.is_setup and self.val_data.is_setup and self.test_data.is_setup

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get train dataloader."""
        return DataLoader(self.train_data, shuffle=True, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader."""
        return DataLoader(self.val_data, shuffle=False, batch_size=self.eval_batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader."""
        return DataLoader(self.test_data, shuffle=False, batch_size=self.eval_batch_size, num_workers=self.num_workers)
