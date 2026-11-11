# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MAD-Sim Data Module.

This module provides a PyTorch Lightning DataModule for the MAD-Sim dataset.
The dataset is downloaded from the project's Google Drive link when missing.

Example:
    Create a MAD-Sim datamodule::

        >>> from anomalib.data import MADSim
        >>> datamodule = MADSim(
        ...     root="./datasets/MAD-Sim",
        ...     category="01Gorilla",
        ... )

Notes:
    The root directory must contain the official category folders::

        MAD-Sim/
            01Gorilla/
                train/good/0.png
                test/good/0.png
                test/Burrs/0.png
                test/Missing/0.png
                test/Stains/0.png
                ground_truth/Burrs/0_mask.png
                ground_truth/Missing/0_mask.png
                ground_truth/Stains/0_mask.png

    The official training and test splits are preserved. Validation uses the
    test set by default; use ``val_split_mode="from_test"`` to hold out validation
    images from that set. There is no official validation directory. Camera
    poses in ``transforms.json`` are not loaded. Category names retain the
    archive's spelling, including ``13Pheonix``.

License:
    The dataset is distributed under CC BY-NC-SA 4.0, as documented in the
    dataset directory description at https://github.com/EricLee0224/PAD.

Reference:
    Zhou et al., "PAD: A Dataset and Benchmark for Pose-agnostic Anomaly
    Detection", NeurIPS 2023. https://arxiv.org/abs/2310.07716.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import URLError

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.mad_sim import CATEGORIES, MADSimDataset
from anomalib.data.utils import DownloadInfo, Split, TestSplitMode, ValSplitMode, download_and_extract
from anomalib.data.utils.path import resolve_path_under_root
from anomalib.utils.path import resolve_dataset_root

DOWNLOAD_INFO = DownloadInfo(
    name="MAD-Sim",
    url="https://drive.usercontent.google.com/download?id=1XlW5v_PCXMH49RSKICkskKjd2A5YDMQq&export=download&confirm=t",
    hashsum="22bff12030d0d53a0f6b698e5dd93080511afed3e1d135c5aaefcdf87725dd78",
    filename="MAD-Sim.zip",
)


class MADSim(AnomalibDataModule):
    """MAD-Sim Lightning Data Module.

    Validation uses the test set by default, preserving the complete official
    test split. Camera poses in ``transforms.json`` are not used.

    Args:
        root (Path | str | None): Path to the root of the dataset.
            Defaults to ``"./datasets/MAD-Sim"``.
        category (str): Category of the MAD-Sim dataset (e.g. ``"01Gorilla"`` or
            ``"02Unicorn"``). Defaults to ``"01Gorilla"``.
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
        Create the datamodule and initialize its datasets::

            >>> datamodule = MADSim(root="./datasets/MAD-Sim", category="01Gorilla")
            >>> datamodule.prepare_data()
            >>> datamodule.setup()
    """

    def __init__(
        self,
        root: Path | str | None = "./datasets/MAD-Sim",
        category: str = "01Gorilla",
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
        if category not in CATEGORIES:
            msg = f"Unknown MAD-Sim category {category!r}. Available categories: {CATEGORIES}"
            raise ValueError(msg)
        if val_split_mode == ValSplitMode.FROM_DIR:
            msg = "MAD-Sim has no official validation directory. Use 'same_as_test' or 'from_test'."
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

        root = resolve_dataset_root(root, "MAD-Sim")
        self.root = Path(root)
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting."""
        self.train_data = MADSimDataset(
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
        )
        self.test_data = MADSimDataset(
            split=Split.TEST,
            root=self.root,
            category=self.category,
        )

    def prepare_data(self) -> None:
        """Download and extract the dataset when the selected category is missing.

        Raises:
            FileNotFoundError: If an existing or downloaded category is incomplete.
        """
        category_root = resolve_path_under_root(self.root, self.category, should_exist=False)
        required = ("train/good", "test", "ground_truth")
        if category_root.exists():
            if not all((category_root / folder).is_dir() for folder in required):
                msg = f"MAD-Sim category is incomplete at {category_root}. Restore its missing folders manually."
                raise FileNotFoundError(msg)
            return

        self.root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.root) as temporary_dir:
            try:
                download_and_extract(Path(temporary_dir), DOWNLOAD_INFO)
            except (URLError, TimeoutError) as exc:
                msg = (
                    "Could not download MAD-Sim. Download the archive manually from "
                    "https://drive.google.com/file/d/1XlW5v_PCXMH49RSKICkskKjd2A5YDMQq/view "
                    f"and extract its category directories into {self.root}."
                )
                raise FileNotFoundError(msg) from exc
            extracted_root = Path(temporary_dir) / "MAD-Sim"
            if not all((extracted_root / self.category / folder).is_dir() for folder in required):
                msg = f"Downloaded MAD-Sim archive is missing required folders for {self.category}."
                raise FileNotFoundError(msg)
            for category in CATEGORIES:
                source = resolve_path_under_root(extracted_root, category, should_exist=False)
                destination = resolve_path_under_root(self.root, category, should_exist=False)
                if source.is_dir() and not destination.exists():
                    source.rename(destination)
