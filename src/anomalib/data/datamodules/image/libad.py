# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LIBAD Data Module.

This module provides a PyTorch Lightning DataModule for the LIBAD dataset.
The dataset will not be downloaded automatically. Users must download it from
Hugging Face (https://huggingface.co/datasets/Evenrose/LIBAD) and extract it.

Example:
    Create a LIBAD datamodule::

        >>> from anomalib.data import LIBAD
        >>> datamodule = LIBAD(
        ...     root="./datasets/LIBAD",
        ...     category="1_wrinkling",
        ...     modality="A"
        ... )

Notes:
    The dataset must be downloaded and extracted manually. The expected
    directory structure is::

        datasets/
        └── LIBAD/
            ├── 1_wrinkling/
            │   ├── normal/
            │   │   ├── A01A.tiff
            │   │   ├── A01B.tiff
            │   │   ├── A01X.tiff
            │   │   └── A01L.tiff
            │   └── anomaly/
            │       ├── ...

License:
    LIBAD dataset is released under the BSD 3-Clause License.

Reference:
    Wenbo Sui and Daniel Lichau and Harold Phelippeau and Zhao Liu. (2026).
    LIBAD: A Multimodal Anomaly Detection Benchmark for Li-Ion Battery Electrode Manufacturing.
"""

import logging
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.libad import LIBADDataset
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode, concatenate_datasets, random_split
from anomalib.utils.path import resolve_dataset_root

logger = logging.getLogger(__name__)


class LIBAD(AnomalibDataModule):
    """LIBAD Lightning Data Module.

    Args:
        root (Path | str | None): Path to the root of the dataset.
            Defaults to ``"./datasets/LIBAD"``.
        category (str): Category of the LIBAD dataset (e.g. ``"1_wrinkling"``).
            Defaults to ``"1_wrinkling"``.
        modality (str): Modality suffix to filter images by.
            Options: 'A' (side A), 'B' (side B), 'L' (X-ray Low), 'X' (X-ray High).
            Defaults to ``"A"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply to the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        test_split_mode (TestSplitMode): Setting that determines how the testing
            subset is obtained.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of images from the train set that will
            be reserved for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Setting that determines how the validation
            subset is obtained.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of train or test images that will be
            reserved for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed which may be set to a fixed value for
            reproducibility.
            Defaults to ``None``.

    Example:
        To create the LIBAD datamodule, instantiate the class and call
        ``setup``::

            >>> from anomalib.data import LIBAD
            >>> datamodule = LIBAD(
            ...     root="./datasets/LIBAD",
            ...     category="1_wrinkling",
            ...     train_batch_size=32,
            ...     eval_batch_size=32,
            ...     num_workers=8,
            ... )
            >>> datamodule.setup()
    """

    def __init__(
        self,
        root: Path | str | None = "./datasets/LIBAD",
        category: str = "1_wrinkling",
        modality: str = "A",
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

        root = resolve_dataset_root(root, "LIBAD")
        self.root = Path(root)
        self.category = category
        self.modality = modality

    def _setup(self, _stage: str | None = None) -> None:
        # Load the full normal dataset (label_index = 0)
        full_normal_dataset = LIBADDataset(
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
            modality=self.modality,
        )

        # Split the normal images into train and test
        self.train_data, normal_test_data = random_split(
            full_normal_dataset,
            split_ratio=[1 - self.test_split_ratio, self.test_split_ratio],
            label_aware=True,
            seed=self.seed,
        )

        # Load the anomalous dataset (label_index = 1)
        anomaly_dataset = LIBADDataset(
            split=Split.TEST,
            root=self.root,
            category=self.category,
            modality=self.modality,
        )

        # Combine normal test and anomalous test data
        self.test_data = concatenate_datasets([normal_test_data, anomaly_dataset])

    def prepare_data(self) -> None:
        """Inform the user to download the dataset if not available.

        This dataset is not automatically downloaded.
        """
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            logger.error(
                "Dataset not found in %s. Please download it from Hugging Face "
                "(https://huggingface.co/datasets/Evenrose/LIBAD) and extract it.",
                self.root / self.category,
            )
            msg = "LIBAD dataset must be downloaded manually."
            raise FileNotFoundError(msg)
