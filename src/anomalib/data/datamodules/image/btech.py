"""BTech Data Module.

This script contains PyTorch Lightning DataModule for the BTech dataset.

If the dataset is not on the file system, the script downloads and
extracts the dataset and create PyTorch data objects.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import shutil
from pathlib import Path

import cv2
from torchvision.transforms.v2 import Transform
from tqdm import tqdm

from anomalib import TaskType
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.btech import BTechDataset
from anomalib.data.utils import (
    DownloadInfo,
    Split,
    TestSplitMode,
    ValSplitMode,
    download_and_extract,
)

logger = logging.getLogger(__name__)

DOWNLOAD_INFO = DownloadInfo(
    name="btech",
    url="https://avires.dimi.uniud.it/papers/btad/btad.zip",
    hashsum="461c9387e515bfed41ecaae07c50cf6b10def647b36c9e31d239ab2736b10d2a",
)


class BTech(AnomalibDataModule):
    """BTech Lightning Data Module.

    Args:
        root (Path | str): Path to the BTech dataset.
            Defaults to ``"./datasets/BTech"``.
        category (str): Name of the BTech category.
            Defaults to ``"01"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Eval batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        task (TaskType, optional): Task type.
            Defaults to ``TaskType.SEGMENTATION``.
        image_size (tuple[int, int], optional): Size to which input images should be resized.
            Defaults to ``None``.
        transform (Transform, optional): Transforms that should be applied to the input images.
            Defaults to ``None``.
        train_transform (Transform, optional): Transforms that should be applied to the input images during training.
            Defaults to ``None``.
        eval_transform (Transform, optional): Transforms that should be applied to the input images during evaluation.
            Defaults to ``None``.
        test_split_mode (TestSplitMode, optional): Setting that determines how the testing subset is obtained.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float, optional): Fraction of images from the train set that will be reserved for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode, optional): Setting that determines how the validation subset is obtained.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float, optional): Fraction of train or test images that will be reserved for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed which may be set to a fixed value for reproducibility.
            Defaults to ``None``.

    Examples:
        To create the BTech datamodule, we need to instantiate the class, and call the ``setup`` method.

        >>> from anomalib.data import BTech
        >>> datamodule = BTech(
        ...     root="./datasets/BTech",
        ...     category="01",
        ...     image_size=256,
        ...     train_batch_size=32,
        ...     eval_batch_size=32,
        ...     num_workers=8,
        ...     transform_config_train=None,
        ...     transform_config_eval=None,
        ... )
        >>> datamodule.setup()

        To get the train dataloader and the first batch of data:

        >>> i, data = next(enumerate(datamodule.train_dataloader()))
        >>> data.keys()
        dict_keys(['image'])
        >>> data["image"].shape
        torch.Size([32, 3, 256, 256])

        To access the validation dataloader and the first batch of data:

        >>> i, data = next(enumerate(datamodule.val_dataloader()))
        >>> data.keys()
        dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
        >>> data["image"].shape, data["mask"].shape
        (torch.Size([32, 3, 256, 256]), torch.Size([32, 256, 256]))

        Similarly, to access the test dataloader and the first batch of data:

        >>> i, data = next(enumerate(datamodule.test_dataloader()))
        >>> data.keys()
        dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
        >>> data["image"].shape, data["mask"].shape
        (torch.Size([32, 3, 256, 256]), torch.Size([32, 256, 256]))
    """

    def __init__(
        self,
        root: Path | str = "./datasets/BTech",
        category: str = "01",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType | str = TaskType.SEGMENTATION,
        image_size: tuple[int, int] | None = None,
        transform: Transform | None = None,
        train_transform: Transform | None = None,
        eval_transform: Transform | None = None,
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
            image_size=image_size,
            transform=transform,
            train_transform=train_transform,
            eval_transform=eval_transform,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.category = category
        self.task = TaskType(task)

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = BTechDataset(
            task=self.task,
            transform=self.train_transform,
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
        )
        self.test_data = BTechDataset(
            task=self.task,
            transform=self.eval_transform,
            split=Split.TEST,
            root=self.root,
            category=self.category,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available.

        This method checks if the specified dataset is available in the file system.
        If not, it downloads and extracts the dataset into the appropriate directory.

        Example:
            Assume the dataset is not available on the file system.
            Here's how the directory structure looks before and after calling the
            `prepare_data` method:

            Before:

            .. code-block:: bash

                $ tree datasets
                datasets
                ├── dataset1
                └── dataset2

            Calling the method:

            .. code-block:: python

                >> datamodule = BTech(root="./datasets/BTech", category="01")
                >> datamodule.prepare_data()

            After:

            .. code-block:: bash

                $ tree datasets
                datasets
                ├── dataset1
                ├── dataset2
                └── BTech
                    ├── 01
                    ├── 02
                    └── 03
        """
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root.parent, DOWNLOAD_INFO)

            # rename folder and convert images
            logger.info("Renaming the dataset directory")
            shutil.move(src=str(self.root.parent / "BTech_Dataset_transformed"), dst=str(self.root))
            logger.info("Convert the bmp formats to png to have consistent image extensions")
            for filename in tqdm(self.root.glob("**/*.bmp"), desc="Converting bmp to png"):
                image = cv2.imread(str(filename))
                cv2.imwrite(str(filename.with_suffix(".png")), image)
                filename.unlink()
