# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Kolektor Surface-Defect Dataset.

Description:
    This module provides a PyTorch Dataset implementation for the Kolektor
    Surface-Defect dataset. The dataset can be accessed at `Kolektor
    Surface-Defect Dataset <https://www.vicos.si/resources/kolektorsdd/>`_.

License:
    The Kolektor Surface-Defect dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0). For more details, visit `Creative Commons License
    <https://creativecommons.org/licenses/by-nc-sa/4.0/>`_.

Reference:
    Tabernik, Domen, Samo Šela, Jure Skvarč, and Danijel Skočaj.
    "Segmentation-based deep-learning approach for surface-defect detection."
    Journal of Intelligent Manufacturing 31, no. 3 (2020): 759-776.
"""

from pathlib import Path

import numpy as np
import polars as pl
from cv2 import imread
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import Split, validate_path
from anomalib.data.utils.dataframe import AnomalibDataFrame


class KolektorDataset(AnomalibDataset):
    """Kolektor dataset class.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/kolektor"``.
        transform (Transform | None, optional): Transforms that should be applied
            to the input images. Defaults to ``None``.
        split (str | Split | None, optional): Split of the dataset, usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import KolektorDataset
        >>> dataset = KolektorDataset(
        ...     root=Path("./datasets/kolektor"),
        ...     split="train"
        ... )
    """

    def __init__(
        self,
        root: Path | str = "./datasets/kolektor",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.root = root
        self.split = split
        self.samples = make_kolektor_dataset(
            self.root,
            train_split_ratio=0.8,
            split=self.split,
        )


def make_kolektor_dataset(
    root: str | Path,
    train_split_ratio: float = 0.8,
    split: str | Split | None = None,
) -> AnomalibDataFrame:
    """Create Kolektor samples by parsing the Kolektor data file structure.

    The files are expected to follow this structure:
        - Image files: ``path/to/dataset/item/image_filename.jpg``
        - Mask files: ``path/to/dataset/item/mask_filename.bmp``

    Example file paths:
        - ``path/to/dataset/kos01/Part0.jpg``
        - ``path/to/dataset/kos01/Part0_label.bmp``

    This function creates a DataFrame with the following columns:
        - ``path``: Base path to dataset
        - ``item``: Item/component name
        - ``split``: Dataset split (train/test)
        - ``label``: Class label (Good/Bad)
        - ``image_path``: Path to image file
        - ``mask_path``: Path to mask file
        - ``label_index``: Numeric label (0=good, 1=bad)

    Args:
        root (str | Path): Path to the dataset root directory.
        train_split_ratio (float, optional): Ratio for splitting good images into
            train/test sets. Defaults to ``0.8``.
        split (str | Split | None, optional): Dataset split (train/test).
            Defaults to ``None``.

    Returns:
        DataFrame: DataFrame containing the dataset samples.

    Example:
        >>> from pathlib import Path
        >>> root = Path('./datasets/kolektor')
        >>> samples = make_kolektor_dataset(root, train_split_ratio=0.8)
        >>> samples.head()
           path     item  split label  image_path              mask_path   label_index
        0  kolektor kos01 train  Good  kos01/Part0.jpg        Part0.bmp   0
        1  kolektor kos01 train  Good  kos01/Part1.jpg        Part1.bmp   0
    """
    root = validate_path(root)

    # Get list of images and masks
    samples_list = [(str(root), *f.parts[-2:]) for f in root.glob(r"**/*") if f.suffix == ".jpg"]
    masks_list = [(str(root), *f.parts[-2:]) for f in root.glob(r"**/*") if f.suffix == ".bmp"]

    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    # Create dataframes
    samples = pl.DataFrame(samples_list, schema=["path", "item", "image_path"], orient="row")
    masks = pl.DataFrame(masks_list, schema=["path", "item", "image_path"], orient="row")

    # Modify image_path column by converting to absolute path
    samples = samples.with_columns(
        (pl.col("path") + "/" + pl.col("item") + "/" + pl.col("image_path")).alias("image_path"),
    )
    masks = masks.with_columns(
        (pl.col("path") + "/" + pl.col("item") + "/" + pl.col("image_path")).alias("image_path"),
    )

    # Sort samples by image path
    samples = samples.sort("image_path")
    masks = masks.sort("image_path")

    # Add mask paths for sample images
    samples = samples.with_columns(masks["image_path"].alias("mask_path"))

    # Use is_good func to configure the label_index
    samples = samples.with_columns(
        pl.col("mask_path").map_elements(is_mask_anomalous, return_dtype=pl.Int64).alias("label_index"),
    )

    # Use label indexes to label data
    samples = samples.with_columns(
        pl.when(pl.col("label_index") == 0).then(pl.lit("Good")).otherwise(pl.lit("Bad")).alias("label"),
    )

    # Add all 'Bad' samples to test set
    samples = samples.with_columns(
        pl.when(pl.col("label") == "Bad").then(pl.lit("test")).otherwise(pl.lit("")).alias("split"),
    )

    # Divide 'good' images to train/test on 0.8/0.2 ratio
    good_samples = samples.filter(pl.col("label") == "Good")
    n_train = int(len(good_samples) * train_split_ratio)
    # Use consistent random split
    rng = np.random.RandomState(42)
    good_indices = rng.permutation(len(good_samples))
    train_indices_good = good_indices[:n_train]
    test_indices_good = good_indices[n_train:]

    # Map back to full dataframe: good samples are those with split == ""
    good_row_indices = samples.with_row_index("_idx").filter(pl.col("label") == "Good")["_idx"].to_list()
    train_global = [good_row_indices[i] for i in train_indices_good]
    test_global = [good_row_indices[i] for i in test_indices_good]

    split_col = samples["split"].to_list()
    for i in train_global:
        split_col[i] = "train"
    for i in test_global:
        split_col[i] = "test"
    samples = samples.with_columns(pl.Series("split", split_col))

    # Reorder columns
    samples = samples.select(["path", "item", "split", "label", "image_path", "mask_path", "label_index"])

    # assert that the right mask files are associated with the right test images
    abnormal = samples.filter(pl.col("label_index") == 1)
    mismatch = False
    for row in abnormal.iter_rows(named=True):
        if Path(row["image_path"]).stem not in Path(row["mask_path"]).stem:
            mismatch = True
            break
    if mismatch:
        msg = """Mismatch between anomalous images and ground truth masks. Make
        sure the mask files follow the same naming convention as the anomalous
        images in the dataset (e.g. image: 'Part0.jpg', mask:
        'Part0_label.bmp')."""
        raise MisMatchError(msg)

    # infer the task type
    task = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    # Get the dataframe for the required split
    if split:
        samples = samples.filter(pl.col("split") == split)

    return AnomalibDataFrame(samples, attrs={"task": task})


def is_mask_anomalous(path: str) -> int:
    """Check if a mask shows defects.

    Args:
        path (str): Path to the mask file.

    Returns:
        int: ``1`` if the mask shows defects, ``0`` otherwise.

    Example:
        >>> from anomalib.data.datasets.image.kolektor import is_mask_anomalous
        >>> path = './datasets/kolektor/kos01/Part0_label.bmp'
        >>> is_mask_anomalous(path)
        1
    """
    img_arr = imread(path)
    if np.all(img_arr == 0):
        return 0
    return 1
