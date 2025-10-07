# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""3D-ADAM Datamodule.

This module provides a PyTorch Lightning DataModule for the 3D-ADAM dataset.
The dataset contains RGB and depth image pairs for anomaly detection tasks.

Example:
    Create a ADAM3D datamodule::

        >>> from anomalib.data import ADAM3D
        >>> datamodule = ADAM3D(
        ...     root="./datasets/ADAM3D",
        ...     category="1m1"
        ... )

License:
    3D-ADAM dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0).
    https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference: https://arxiv.org/abs/2507.07838

"""

import logging
from pathlib import Path
from shutil import move

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.depth.adam_3d import ADAM3DDataset
from anomalib.data.utils import DownloadInfo, Split, TestSplitMode, ValSplitMode, download_and_extract

logger = logging.getLogger(__name__)


DOWNLOAD_INFO = DownloadInfo(
    name="adam_3d",
    url="https://huggingface.co/datasets/pmchard/3D-ADAM_anomalib/resolve/main/adam3d_cropped.zip",
    hashsum="ffc4c52afa1566a4165c42300c21758ec8292ff04305c65e81e830abb8200c36",
)


class ADAM3D(AnomalibDataModule):
    """3D-ADAM Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/ADAM3D"``.
        category (str): Category of the 3D-ADAM dataset (e.g. ``"1m1"`` or
            ``"spiral_gear"``). Defaults to ``"1m1"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers for data loading.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply dto the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        test_split_mode (TestSplitMode | str): Method to create test set.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of data to use for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode | str): Method to create validation set.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of data to use for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Random seed for reproducibility.
            Defaults to ``None``.
    """

    def __init__(
        self,
        root: Path | str = "./datasets/ADAM3D",
        category: str = "1m1",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets.

        Args:
            _stage (str | None, optional): Stage of setup. Not used.
                Defaults to ``None``.
        """
        self.train_data = ADAM3DDataset(
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
        )
        self.test_data = ADAM3DDataset(
            split=Split.TEST,
            root=self.root,
            category=self.category,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available."""
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)
            # The Huggingface dataset is stored in adam3d_cropped
            # Move the contents to the root
            extracted_folder = self.root / "adam3d_cropped"
            for filename in extracted_folder.glob("*"):
                move(str(filename), str(self.root / filename.name))
            extracted_folder.rmdir()
