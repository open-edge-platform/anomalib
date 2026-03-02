# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""BMAD Data Module.

This module provides a PyTorch Lightning DataModule for the BMAD (Benchmarks for Medical Anomaly Detection) dataset.
If the dataset is not available locally, it will be downloaded and prepared automatically.

BMAD is a standardized benchmark comprising six reorganized public medical-imaging datasets from five domains:
brain MRI, liver CT, retinal OCT, chest X-ray, and digital histopathology.
Within these datasets, three support pixel-level anomaly localization,
while the remaining three are for sample-level anomaly detection only :contentReference[oaicite:0]{index=0}.

Example:
    Create a BMAD datamodule::

        >>> from anomalib.data import BMAD
        >>> datamodule = BMAD(
        ...     root="./datasets/BMAD",
        ...     dataset="Brain",       # options: "Brain", "Chest", "Histopathology", "Liver", "Retina_OCT2017",
                                                    "Retina_RESC"
        ... )

Notes:
    The dataset will be automatically downloaded and reorganized upon first usage.
    Directory structure after preparation may look like:

    .. code-block:: text

        datasets/
        └── BMAD/
            ├── Brain/
            │   ├── train/
            │   │   └── good/
            │   ├── valid/
            │   │   ├── good/
            │   │   └── Ungood/ (if applicable with masks for localization)
            │   └── test/
            │       ├── good/
            │       └── Ungood/
            ├── Liver/
            ├── Retina_OCT2017/
            ├── Retina_RESC/
            ├── Chest/
            └── Histopathology/

License:
    BMAD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0).
    https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Jinan Bao, Hanshi Sun, Hanqiu Deng, Yinsheng He, Zhaoxiang Zhang, Xingyu Li:
    "BMAD: Benchmarks for Medical Anomaly Detection," arXiv preprint arXiv:2306.11876, 2023.
    DOI: 10.48550/arXiv.2306.11876
    https://arxiv.org/abs/2306.11876
"""

import logging
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.bmad import BMADDataset
from anomalib.data.utils import DownloadInfo, Split, TestSplitMode, ValSplitMode, download_and_extract

logger = logging.getLogger(__name__)

DOWNLOAD_INFO = DownloadInfo(
    name="bmad",
    url="https://huggingface.co/datasets/code-dev05/BMAD/resolve/main/bmad.zip",
    hashsum="df655def31f3f638a91c567550de54d6e45a74b2368f666c13a6a3052c063165",
)


class BMAD(AnomalibDataModule):
    """BMAD Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/BMAD"``.
        category (str): Category of the BMAD dataset
            (e.g. ``"Brain"``, ``"Liver"``, ``"Retina_OCT2017"``, ``"Retina_RESC"``,
            ``"Chest"``, or ``"Histopathology"``).
            Defaults to ``"Brain"``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply to the training images.
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
        seed (int | None, optional): Seed for reproducibility.
            Defaults to ``None``.

    Example:
        Create BMAD datamodule with default settings::

            >>> datamodule = BMAD()
            >>> datamodule.setup()
            >>> i, data = next(enumerate(datamodule.train_dataloader()))

            >>> data.image.shape
            torch.Size([32, 3, 240, 240])

        Change the category::

            >>> datamodule = BMAD(category="Liver")

        Use Retina_RESC::

            >>> datamodule = BMAD(category="Retina_RESC")

        Create validation set from test data::

            >>> datamodule = BMAD(
            ...     val_split_mode=ValSplitMode.FROM_TEST,
            ...     val_split_ratio=0.1
            ... )
    """

    def __init__(
        self,
        root: Path | str = "./datasets/BMAD",
        category: str = "Brain",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.FROM_DIR,
        val_split_ratio: float | None = None,
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
        self.train_data = BMADDataset(
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
        )
        if self.val_split_mode == ValSplitMode.FROM_DIR:
            self.val_data = BMADDataset(
                split="valid",
                root=self.root,
                category=self.category,
            )
        self.test_data = BMADDataset(
            split=Split.TEST,
            root=self.root,
            category=self.category,
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available.

        This method checks if the specified dataset is available in the file
        system. If not, it downloads and extracts the dataset into the
        appropriate directory.

        Example:
            Assume the dataset is not available on the file system::

                >>> datamodule = BMAD(
                ...     root="./datasets/BMAD",
                ...     category="Brain"
                ... )
                >>> datamodule.prepare_data()

            Directory structure after download::

                datasets/
                └── BMAD/
                    ├── Brain/
                    ├── Liver/
                    └── ...
        """
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)
