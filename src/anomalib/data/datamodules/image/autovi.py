# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AutoVI Data Module.

This module provides a PyTorch Lightning DataModule for the AutoVI
(Automotive Visual Inspection) dataset.

The dataset is hosted on Zenodo as **six independent zip files** — one per
class.  When :meth:`AutoVI.prepare_data` is called, only the zip file that
corresponds to the requested ``category`` is downloaded and extracted, so you
don't have to fetch the full 4 GB archive if you only need one class.

Usage::

    from anomalib.data.datamodules.image import AutoVI

    datamodule = AutoVI(
        root="./datasets/AutoVI",
        category="pipe_clip",
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8,
    )
    datamodule.setup()

    for batch in datamodule.train_dataloader():
        print(batch["image"].shape)   # (32, 3, H, W)

Data License:
    Copyright © 2023-2024 Renault Group.
    Released under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
    International (CC BY-NC-SA 4.0).
    https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Carvalho, P., Lafou, M., Durupt, A., Leblanc, A., & Grandvalet, Y. (2024).
    The Automotive Visual Inspection Dataset (AutoVI): A Genuine Industrial
    Production Dataset for Unsupervised Anomaly Detection [Dataset].
    https://doi.org/10.5281/zenodo.10459003
"""

from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.autovi import (
    CATEGORIES,
    DOWNLOAD_HASHES,
    DOWNLOAD_URLS,
    AutoVIDataset,
)
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode
from anomalib.data.utils.download import DownloadInfo, download_and_extract
from anomalib.utils.path import get_datasets_dir


class AutoVI(AnomalibDataModule):
    """AutoVI Lightning DataModule.

    Downloads (on demand) and wraps the AutoVI dataset for use with the
    anomalib training engine.  Because the dataset ships as **six separate
    zip files**, only the zip that belongs to the requested ``category`` is
    fetched.

    The resulting on-disk layout mirrors MVTec AD, except that ground-truth
    masks are stored one level deeper — each mask lives in a subdirectory
    named after the corresponding test image stem::

        datasets/
        └── AutoVI/
            ├── engine_wiring/
            │   ├── train/good/*.png
            │   ├── test/{good,<defect_type>}/*.png
            │   └── ground_truth/<defect_type>/<image_stem>/0000.png
            ├── pipe_clip/
            ├── pipe_staple/
            ├── tank_screw/
            ├── underbody_pipes/
            └── underbody_screw/

    For example, the mask for ``test/unclipped/0089.png`` is stored at
    ``ground_truth/unclipped/0089/0000.png``.

    Args:
        root (Path | str | None): Root directory that will contain (or already
            contains) the per-category sub-directories.
            Defaults to ``None`` → ``<anomalib_datasets_dir>/AutoVI``.
        category (str): One of the six AutoVI classes.
            Defaults to ``"engine_wiring"``.
        train_batch_size (int): Training batch size.  Defaults to ``32``.
        eval_batch_size (int): Validation / test batch size.  Defaults to ``32``.
        num_workers (int): Number of DataLoader workers.  Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations for the training
            split.  Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations for the validation
            split.  Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations for the test split.
            Defaults to ``None``.
        augmentations (Transform | None): Fallback augmentations used when no
            split-specific augmentation is supplied.  Defaults to ``None``.
        test_split_mode (TestSplitMode): How to obtain the test set.
            Defaults to :attr:`TestSplitMode.FROM_DIR`.
        test_split_ratio (float): Fraction of data to use for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): How to obtain the validation set.
            Defaults to :attr:`ValSplitMode.SAME_AS_TEST`.
        val_split_ratio (float): Fraction of data to use for validation.
            Defaults to ``0.5``.
        seed (int | None): Random seed for reproducibility.
            Defaults to ``None``.

    Example:
        >>> datamodule = AutoVI(
        ...     root="./datasets/AutoVI",
        ...     category="pipe_clip",
        ... )
        >>> datamodule.setup()
        >>> i, batch = next(enumerate(datamodule.train_dataloader()))
        >>> batch["image"].shape
        torch.Size([32, 3, 256, 256])
    """

    def __init__(
        self,
        root: Path | str | None = None,
        category: str = "engine_wiring",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if category not in CATEGORIES:
            msg = f"Invalid category '{category}'. Choose from: {CATEGORIES}"
            raise ValueError(msg)

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

        self.root = Path(root) if root is not None else get_datasets_dir() / "AutoVI"
        self.category = category

    def _setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Instantiate train and test datasets for the chosen category."""
        self.train_data = AutoVIDataset(
            root=self.root,
            category=self.category,
            augmentations=self.train_augmentations,
            split=Split.TRAIN,
        )
        self.test_data = AutoVIDataset(
            root=self.root,
            category=self.category,
            augmentations=self.test_augmentations,
            split=Split.TEST,
        )

    def prepare_data(self) -> None:
        """Download and extract the zip for the requested category, if needed.

        Only the single zip that belongs to ``self.category`` is fetched from
        Zenodo — the remaining five classes are left untouched.

        The check is: if ``<root>/<category>/`` already exists as a directory
        we assume the data are present and skip the download.

        Example:
            >>> datamodule = AutoVI(
            ...     root="./datasets/AutoVI",
            ...     category="tank_screw",
            ... )
            >>> datamodule.prepare_data()   # downloads tank_screw.zip only

        Directory structure after a successful download::

                datasets/
                    └── AutoVI/
                        ├── engine_wiring/
                        │   ├── train/good/*.png
                        │   ├── test/{good,<defect_type>}/*.png
                        │   └── ground_truth/<defect_type>/<image_stem>/0000.png
        """
        category_dir = self.root / self.category
        if category_dir.is_dir():
            return

        self.root.mkdir(parents=True, exist_ok=True)

        download_info = DownloadInfo(
            name=f"AutoVI - {self.category}",
            url=DOWNLOAD_URLS[self.category],
            # The zip extracts to a folder named after the category, so we
            # extract directly into `self.root` and end up with
            # `self.root/<category>/...`
            filename=f"{self.category}.zip",
            hashsum=DOWNLOAD_HASHES[self.category],
        )
        download_and_extract(self.root, download_info)
