# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Kaputt Data Module.

This module provides a PyTorch Lightning DataModule for the Kaputt dataset.

The Kaputt dataset is a large-scale dataset for visual defect detection in
logistics settings. It has more than 230,000 images with greater than 29,000
defective instances, and 48,000 distinct objects.

Example:
    Create a Kaputt datamodule::

        >>> from anomalib.data import Kaputt
        >>> datamodule = Kaputt(
        ...     root="./datasets/kaputt",
        ... )

Notes:
    The Kaputt dataset requires manual download from
    https://www.kaputt-dataset.com/. Please fill out the form on the website
    to obtain access to the dataset.

    The expected directory structure after extraction is::

        datasets/kaputt/
        ├── datasets/                         # Parquet metadata files
        │   ├── query-train.parquet
        │   ├── query-validation.parquet
        │   ├── query-test.parquet
        │   ├── reference-train.parquet
        │   ├── reference-validation.parquet
        │   └── reference-test.parquet
        │
        ├── query-image/data/<split>/query-data/image/   # Query images
        ├── query-crop/data/<split>/query-data/crop/     # Cropped regions
        ├── query-mask/data/<split>/query-data/mask/     # Segmentation masks
        │
        ├── reference-image/data/<split>/reference-data/image/
        ├── reference-crop/data/<split>/reference-data/crop/
        └── reference-mask/data/<split>/reference-data/mask/

License:
    The Kaputt dataset is released under the Creative Commons
    Attribution-NonCommercial-NoDerivatives 4.0 International License
    (CC BY-NC-ND 4.0).
    https://creativecommons.org/licenses/by-nc-nd/4.0/

Reference:
    Höfer, S., Henning, D., Amiranashvili, A., Morrison, D., Tzes, M.,
    Posner, I., Matvienko, M., Rennola, A., & Milan, A. (2025).
    Kaputt: A Large-Scale Dataset for Visual Defect Detection.
    In IEEE/CVF International Conference on Computer Vision (ICCV).
"""

import logging
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.kaputt import KaputtDataset
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode

logger = logging.getLogger(__name__)


class Kaputt(AnomalibDataModule):
    """Kaputt Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/kaputt"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        image_type (str): Type of images to use - "image" for full images or
            "crop" for cropped item regions. Defaults to ``"image"``.
        use_reference (bool): If True, include reference (defect-free) images
            in addition to query images. Reference images are always labeled
            as normal. Defaults to ``False``.
        train_augmentations (Transform | None): Augmentations to apply to the training images.
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        test_split_mode (TestSplitMode): Method to create test set.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of data to use for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Method to create validation set.
            Defaults to ``ValSplitMode.FROM_DIR`` since Kaputt has native validation split.
        val_split_ratio (float): Fraction of data to use for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed for reproducibility.
            Defaults to ``None``.

    Example:
        Create Kaputt datamodule with default settings::

            >>> datamodule = Kaputt()
            >>> datamodule.setup()
            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data.keys()
            dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

            >>> data["image"].shape
            torch.Size([32, 3, 256, 256])

        Use cropped images instead of full images::

            >>> datamodule = Kaputt(image_type="crop")

        Include reference (defect-free) images in training::

            >>> datamodule = Kaputt(use_reference=True)

        Create validation set from test data (instead of using native val split)::

            >>> datamodule = Kaputt(
            ...     val_split_mode=ValSplitMode.FROM_TEST,
            ...     val_split_ratio=0.1
            ... )

    Note:
        The Kaputt dataset must be downloaded manually from
        https://www.kaputt-dataset.com/. This datamodule does not support
        automatic download due to the dataset's licensing requirements.
    """

    def __init__(
        self,
        root: Path | str = "./datasets/kaputt",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        image_type: str = "image",
        use_reference: bool = False,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.FROM_DIR,
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
        self.image_type = image_type
        self.use_reference = use_reference

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Note:
            The Kaputt dataset has native train/validation/test splits, so we
            use them directly when val_split_mode is FROM_DIR.
        """
        self.train_data = KaputtDataset(
            split=Split.TRAIN,
            root=self.root,
            image_type=self.image_type,
            use_reference=self.use_reference,
        )
        self.test_data = KaputtDataset(
            split=Split.TEST,
            root=self.root,
            image_type=self.image_type,
            use_reference=False,  # Don't use reference for test
        )

        # Kaputt has a native validation split
        if self.val_split_mode == ValSplitMode.FROM_DIR:
            self.val_data = KaputtDataset(
                split=Split.VAL,
                root=self.root,
                image_type=self.image_type,
                use_reference=False,  # Don't use reference for validation
            )

    def prepare_data(self) -> None:
        """Check if the dataset is available.

        This method checks if the specified dataset is available in the file
        system. Unlike other datasets, Kaputt does not support automatic
        download due to licensing requirements.

        Raises:
            FileNotFoundError: If the dataset is not found at the specified path.

        Example:
            Assume the dataset is available on the file system::

                >>> datamodule = Kaputt(
                ...     root="./datasets/kaputt",
                ... )
                >>> datamodule.prepare_data()

            Directory structure should include::

                datasets/kaputt/
                ├── datasets/
                │   ├── query-train.parquet
                │   ├── query-validation.parquet
                │   └── query-test.parquet
                └── query-image/
                    └── data/
                        ├── train/
                        ├── validation/
                        └── test/
        """
        datasets_dir = self.root / "datasets"
        query_train_parquet = datasets_dir / "query-train.parquet"

        if datasets_dir.is_dir() and query_train_parquet.exists():
            logger.info("Found the Kaputt dataset.")
        else:
            msg = (
                f"Kaputt dataset not found at {self.root}. "
                "Please download the dataset from https://www.kaputt-dataset.com/ "
                "and extract it to the specified root directory. "
                "Note: The dataset requires filling out a request form for access. "
                f"Expected to find: {query_train_parquet}"
            )
            raise FileNotFoundError(msg)
