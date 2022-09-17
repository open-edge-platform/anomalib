"""Anomalib dataset and datamodule base classes."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
from pandas import DataFrame
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from anomalib.data.utils import read_image
from anomalib.pre_processing import PreProcessor

logger = logging.getLogger(__name__)


class AnomalibDataset(Dataset):
    """Anomalib dataset."""

    def __init__(self, samples: DataFrame, task: str, split: str, pre_process: PreProcessor):
        super().__init__()
        self.samples = samples
        self.task = task
        self.split = split
        self.pre_process = pre_process

    def __len__(self) -> int:
        """Get length of the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[str, Tensor]]:
        """Get dataset item for the index ``index``.

        Args:
            index (int): Index to get the item.

        Returns:
            Union[Dict[str, Tensor], Dict[str, Union[str, Tensor]]]: Dict of image tensor during training.
                Otherwise, Dict containing image path, target path, image tensor, label and transformed bounding box.
        """
        image_path = self.samples.iloc[index].image_path
        image = read_image(image_path)
        label_index = self.samples.iloc[index].label_index

        item = dict(image_path=image_path, label=label_index)

        if self.task == "classification":
            pre_processed = self.pre_process(image=image)
        elif self.task == "segmentation":
            mask_path = self.samples.iloc[index].mask_path

            # Only Anomalous (1) images have masks in anomaly datasets
            # Therefore, create empty mask for Normal (0) images.
            if label_index == 0:
                mask = np.zeros(shape=image.shape[:2])
            else:
                mask = cv2.imread(mask_path, flags=0) / 255.0

            pre_processed = self.pre_process(image=image, mask=mask)

            item["mask_path"] = mask_path
            item["mask"] = pre_processed["mask"]
        else:
            raise ValueError(f"Unknown task type: {self.task}")
        item["image"] = pre_processed["image"]

        return item


class AnomalibDataModule(LightningDataModule, ABC):
    """Base Anomalib data module."""

    def __init__(
        self,
        task: str,
        train_batch_size: int,
        test_batch_size: int,
        num_workers: int,
        transform_config_train: Optional[Union[str, A.Compose]] = None,
        transform_config_val: Optional[Union[str, A.Compose]] = None,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        create_validation_set: bool = False,
    ):
        super().__init__()
        self.task = task
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.create_validation_set = create_validation_set

        if transform_config_train is not None and transform_config_val is None:
            transform_config_val = transform_config_train
        self.pre_process_train = PreProcessor(config=transform_config_train, image_size=image_size)
        self.pre_process_val = PreProcessor(config=transform_config_val, image_size=image_size)

        self.train_data: Optional[AnomalibDataset] = None
        self.val_data: Optional[AnomalibDataset] = None
        self.test_data: Optional[AnomalibDataset] = None

        self._samples: Optional[DataFrame] = None

    @abstractmethod
    def _create_samples(self) -> DataFrame:
        """This method should be implemented in the subclass.

        This method should return a dataframe that contains the information needed by the dataloader to load each of
        the dataset items into memory. The dataframe must at least contain the following columns:
        split - The subset to which the dataset item is assigned.
        image_path - Path to file system location where the image is stored.
        label_index - Index of the anomaly label, typically 0 for "normal" and 1 for "anomalous".

        Additionally, when the task type is segmentation, the dataframe must have the mask_path column, which contains
        the path the ground truth masks (for the anomalous images only).

        Example of a dataframe returned by calling this method from a concrete class:
        |---|-------------------|-----------|-------------|------------------|-------|
        |   | image_path        | label     | label_index | mask_path        | split |
        |---|-------------------|-----------|-------------|------------------|-------|
        | 0 | path/to/image.png | anomalous | 0           | path/to/mask.png | train |
        |---|-------------------|-----------|-------------|------------------|-------|
        """
        raise NotImplementedError

    def get_samples(self, split: Optional[str] = None) -> DataFrame:
        """Retrieve the samples of the full dataset or one of the splits (train, val, test).

        Args:
            split: (str): The split for which we want to retrieve the samples ("train", "val" or "test"). When
                left empty, all samples will be returned.

        Returns:
            DataFrame: A dataframe containing the samples of the split or full dataset.
        """
        assert self._samples is not None, "Samples have not been created yet."
        if split is None:
            return self._samples
        samples = self._samples[self._samples.split == split]
        return samples.reset_index(drop=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup train, validation and test data.

        Args:
          stage: Optional[str]:  Train/Val/Test stages. (Default value = None)
        """
        self._samples = self._create_samples()

        logger.info("Setting up train, validation, test and prediction datasets.")
        if stage in (None, "fit"):
            samples = self.get_samples("train")
            self.train_data = AnomalibDataset(
                samples=samples,
                split="train",
                task=self.task,
                pre_process=self.pre_process_train,
            )

        if stage in (None, "fit", "validate"):
            samples = self.get_samples("val") if self.create_validation_set else self.get_samples("test")
            self.val_data = AnomalibDataset(
                samples=samples,
                split="val",
                task=self.task,
                pre_process=self.pre_process_val,
            )

        if stage in (None, "test"):
            samples = self.get_samples("test")
            self.test_data = AnomalibDataset(
                samples=samples,
                split="test",
                task=self.task,
                pre_process=self.pre_process_val,
            )

    def contains_anomalous_images(self, split: Optional[str] = None) -> bool:
        """Check if the dataset or the specified subset contains any anomalous images.

        Args:
            split (str): the subset of interest ("train", "val" or "test"). When left empty, the full dataset will be
                checked.

        Returns:
            bool: Boolean indicating if any anomalous images have been assigned to the dataset or subset.
        """
        samples = self.get_samples(split)
        return 1 in list(samples.label_index)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get train dataloader."""
        return DataLoader(self.train_data, shuffle=True, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader."""
        return DataLoader(self.val_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader."""
        return DataLoader(self.test_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)
