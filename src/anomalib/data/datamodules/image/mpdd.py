# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MPDD Data Module.

This module provides a PyTorch Lightning DataModule for the MPDD dataset.

MPDD is a dataset aimed at benchmarking visual defect detection methods
in industrial metal parts manufacturing. It contains 6 categories of industrial
objects with both normal and anomalous samples. Each category includes RGB
images and pixel-level ground truth masks for anomaly segmentation.

Example:
    Create a MPDD datamodule::

        >>> from anomalib.data import MPDD
        >>> datamodule = MPDD(
        ...     root="./datasets/MPDD",
        ...     category="bracket_black"
        ... )

Notes:
    The dataset should be downloaded manually from OneDrive and placed in the
    appropriate directory. See ``DOWNLOAD_INSTRUCTIONS`` for detailed steps.

License:
    MPDD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    S. Jezek, M. Jonak, R. Burget, P. Dvorak and M. Skotak (2021).
    Deep learning-based defect detection of metal parts: evaluating
    current methods in complex conditions. 13th International Congress
    on Ultra Modern Telecommunications and Control Systems and Workshops
    (ICUMT), 2021, pp. 66-71, DOI: 10.1109/ICUMT54235.2021.9631567.
"""

import logging
from pathlib import Path
from textwrap import dedent

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.mpdd import MPDDDataset
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode

logger = logging.getLogger(__name__)


class MPDD(AnomalibDataModule):
    """MPDD Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/MPDD"``.
        category (str): Category of the MPDD dataset (e.g. ``"bracket_black"`` or
            ``"bracket_brown"``). Defaults to ``"bracket_black"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply dto the training images
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
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of data to use for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed for reproducibility.
            Defaults to ``None``.

    Example:
        Create MPDD datamodule with default settings::

            >>> datamodule = MPDD()
            >>> datamodule.setup()
            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data.keys()
            dict_keys(['image_path', 'label', 'image', 'mask_path', 'mask'])

            >>> data["image"].shape
            torch.Size([32, 3, 256, 256])

        Change the category::

            >>> datamodule = MPDD(category="bracket_brown")

        Create validation set from test data::

            >>> datamodule = MPDD(
            ...     val_split_mode=ValSplitMode.FROM_TEST,
            ...     val_split_ratio=0.1
            ... )

        Create synthetic validation set::

            >>> datamodule = MPDD(
            ...     val_split_mode=ValSplitMode.SYNTHETIC,
            ...     val_split_ratio=0.2
            ... )
    """

    def __init__(
        self,
        root: Path | str = "./datasets/MPDD",
        category: str = "bracket_black",
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
        """Set up the datasets and perform dynamic subset splitting."""
        self.train_data = MPDDDataset(
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
        )
        self.test_data = MPDDDataset(
            split=Split.TEST,
            root=self.root,
            category=self.category,
        )

    def prepare_data(self) -> None:
        """Verify that the dataset is available and provide download instructions.

        This method checks if the dataset exists in the root directory. If not, it provides
        instructions for downloading from OneDrive.

        The MPDD dataset is available at:
        https://vutbr-my.sharepoint.com/:f:/g/personal/xjezek16_vutbr_cz/EhHS_ufVigxDo3MC6Lweau0BVMuoCmhMZj6ddamiQ7-FnA?e=oHKCxI

        """
        required_dir = Path(self.root)

        if not required_dir.exists():
            raise RuntimeError(get_download_instructions(required_dir))


def get_download_instructions(root_path: Path) -> str:
    """Get download instructions for the MPDD dataset.

    Args:
        root_path: Path where the dataset should be downloaded.

    Returns:
         str: Formatted download instructions.
    """
    return dedent(f"""
        MPDD dataset not found in {root_path}

        The MPDD dataset requires manual download from OneDrive.
        Follow these steps to download and prepare the dataset:
        -----------------------
        a. Visit https://vutbr-my.sharepoint.com/:f:/g/personal/xjezek16_vutbr_cz/EhHS_ufVigxDo3MC6Lweau0BVMuoCmhMZj6ddamiQ7-FnA?e=oHKCxI
        b. Download all files manually
        c. Extract the contents to: {root_path}

        Expected directory structure:
        {root_path}/
            ├── bracket_black/
            ├── bracket_brown/
            └── ...
    """)
