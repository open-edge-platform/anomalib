# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom Tabular Dataset.

This module provides a custom PyTorch Dataset implementation for loading
images using a selection of paths and labels defined in a table or tabular file.
It does not require a specific folder structure and allows subsampling and
relabeling without moving files. The dataset supports both classification and
segmentation tasks.

The table should contain columns for ``image_paths``, ``label_index``, ``split``,
and optionally ``masks_paths`` for segmentation tasks.

Example:
    >>> from anomalib.data.datasets import TabularDataset
    >>> samples = {
    ...     "image_path": ["images/image1.png", "images/image2.png", "images/image3.png", ... ],
    ...     "label_index": [LabelName.NORMAL, LabelName.NORMAL, LabelName.ABNORMAL,  ... ],
    ...     "split": [Split.TRAIN, Split.TRAIN, Split.TEST, ... ],
    ... }
    >>> dataset = TabularDataset(
    ...     name="custom",
    ...     samples=samples,
    ...     root="./datasets/custom",
    ... )
"""

from pathlib import Path

import polars as pl
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import DirType, LabelName, Split
from anomalib.data.utils.dataframe import AnomalibDataFrame


class TabularDataset(AnomalibDataset):
    """Dataset class for loading images from paths and labels defined in a table.

    Args:
        name (str): Name of the dataset. Used for logging/saving.
        samples (dict | list | DataFrame): Pandas ``DataFrame`` or compatible ``list``
            or ``dict`` containing the dataset information.
        augmentations (Transform | None, optional): Augmentations to apply to the images.
            Defaults to ``None``.
        root (str | Path | None, optional): Root directory of the dataset.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split to load.
            Choose from ``Split.FULL``, ``Split.TRAIN``, ``Split.TEST``.
            Defaults to ``None``.

    Examples:
        Create a classification dataset:

        >>> from anomalib.data.utils import InputNormalizationMethod, get_transforms
        >>> from anomalib.data.datasets import TabularDataset
        >>> transform = get_transforms(
        ...     image_size=256,
        ...     normalization=InputNormalizationMethod.NONE
        ... )
        >>> samples = {
        ...     "image_path": ["images/image1.png", "images/image2.png", "images/image3.png", ... ],
        ...     "label_index": [LabelName.NORMAL, LabelName.NORMAL, LabelName.ABNORMAL,  ... ],
        ...     "split": [Split.TRAIN, Split.TRAIN, Split.TEST, ... ],
        ... }
        >>> dataset = TabularDataset(
        ...     name="custom",
        ...     samples=samples,
        ...     root="./datasets/custom",
        ...     transform=transform
        ... )

        Create a segmentation dataset:

        >>> samples = {
        ...     "image_path": ["images/image1.png", "images/image2.png", "images/image3.png", ... ],
        ...     "label_index": [LabelName.NORMAL, LabelName.NORMAL, LabelName.ABNORMAL,  ... ],
        ...     "split": [Split.TRAIN, Split.TRAIN, Split.TEST, ... ],
        ...     "mask_path": ["masks/mask1.png", "masks/mask2.png", "masks/mask3.png", ... ],
        ... }
        >>> dataset = TabularDataset(
        ...     name="custom",
        ...     samples=samples,
        ...     root="./datasets/custom",
        ...     transform=transform
        ... )
    """

    def __init__(
        self,
        name: str,
        samples: dict | list | pl.DataFrame | AnomalibDataFrame,
        augmentations: Transform | None = None,
        root: str | Path | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self._name = name
        self.split = split
        self.root = root
        self.samples = make_tabular_dataset(
            samples=samples,
            root=self.root,
            split=self.split,
        )

    @property
    def name(self) -> str:
        """Get dataset name.

        Returns:
            str: Name of the dataset
        """
        return self._name


def _infer_missing_columns(samples: pl.DataFrame) -> pl.DataFrame:
    """Infer missing label_index, label, and split columns from those present.

    Args:
        samples (pl.DataFrame): DataFrame that must already have ``image_path``.

    Returns:
        pl.DataFrame: DataFrame with ``label_index``, ``label``, ``split`` and
            ``mask_path`` columns guaranteed to exist.
    """
    if "label_index" in samples.columns:
        samples = samples.with_columns(pl.col("label_index").cast(pl.Int64))

    columns_present = [col in samples.columns for col in ["label_index", "label", "split"]]

    # all columns missing
    if columns_present == [False, False, False]:
        msg = "The samples table must contain at least one of 'label_index', 'label' or 'split' columns."
        raise ValueError(msg)

    # label_index missing (split can be present or missing)
    if columns_present[:2] == [False, True]:
        label_to_label_index = {
            str(DirType.ABNORMAL): int(LabelName.ABNORMAL),
            str(DirType.NORMAL): int(LabelName.NORMAL),
            str(DirType.NORMAL_TEST): int(LabelName.NORMAL),
        }
        label_index_col = samples["label"].map_elements(
            label_to_label_index.get,
            return_dtype=pl.Int64,
        )
        samples = samples.with_columns(label_index_col.alias("label_index"))

    # label_index and label missing
    elif columns_present == [False, False, True]:
        split_to_label_index = {
            Split.TRAIN.value: int(LabelName.NORMAL),
            Split.TEST.value: int(LabelName.ABNORMAL),
        }
        label_index_col = samples["split"].map_elements(
            split_to_label_index.get,
            return_dtype=pl.Int64,
        )
        samples = samples.with_columns(label_index_col.alias("label_index"))

    # label and split missing
    elif columns_present == [True, False, False]:
        label_index_to_label = {
            int(LabelName.ABNORMAL): str(DirType.ABNORMAL),
            int(LabelName.NORMAL): str(DirType.NORMAL),
        }
        label_col = samples["label_index"].map_elements(
            label_index_to_label.get,
            return_dtype=pl.Utf8,
        )
        samples = samples.with_columns(label_col.alias("label"))

    # reevaluate columns_present in case a column was added
    columns_present = [col in samples.columns for col in ["label_index", "label", "split"]]

    # label missing
    if columns_present == [True, False, True]:
        samples = samples.with_columns(
            pl.when((pl.col("label_index") == int(LabelName.NORMAL)) & (pl.col("split") == Split.TRAIN.value))
            .then(pl.lit(str(DirType.NORMAL)))
            .when((pl.col("label_index") == int(LabelName.NORMAL)) & (pl.col("split") == Split.TEST.value))
            .then(pl.lit(str(DirType.NORMAL_TEST)))
            .when(pl.col("label_index") == int(LabelName.ABNORMAL))
            .then(pl.lit(str(DirType.ABNORMAL)))
            .otherwise(pl.lit(None).cast(pl.Utf8))
            .alias("label"),
        )
    # split missing
    elif columns_present == [True, True, False]:
        label_to_split = {
            str(DirType.NORMAL): Split.TRAIN.value,
            str(DirType.ABNORMAL): Split.TEST.value,
            str(DirType.NORMAL_TEST): Split.TEST.value,
        }
        split_col = samples["label"].map_elements(
            label_to_split.get,
            return_dtype=pl.Utf8,
        )
        samples = samples.with_columns(split_col.alias("split"))

    # Add mask_path column if not exists
    if "mask_path" not in samples.columns:
        samples = samples.with_columns(pl.lit("").alias("mask_path"))

    return samples


def make_tabular_dataset(
    samples: dict | list | pl.DataFrame | AnomalibDataFrame,
    root: str | Path | None = None,
    split: str | Split | None = None,
) -> AnomalibDataFrame:
    """Create a dataset from a table of image paths and labels.

    Args:
        samples (dict | list | DataFrame): Pandas ``DataFrame`` or compatible
            ``list`` or ``dict`` containing the dataset information.
        root (str | Path | None, optional): Root directory of the dataset.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split to load.
            Choose from ``Split.FULL``, ``Split.TRAIN``, ``Split.TEST``.
            Defaults to ``None``.

    Returns:
        DataFrame: Dataset samples with columns for image paths, labels, splits
            and mask paths (for segmentation).

    Examples:
        Create a classification dataset:
        >>> samples = {
        ...     "image_path": ["images/00.png", "images/01.png", "images/02.png", ... ],
        ...     "label_index": [LabelName.NORMAL, LabelName.NORMAL, LabelName.NORMAL,  ... ],
        ...     "split": [Split.TRAIN, Split.TRAIN, Split.TRAIN, ... ],
        ... }
        >>> tabular_df = make_tabular_dataset(
        ...     samples=samples,
        ...     root="./datasets/custom",
        ...     split=Split.TRAIN,
        ... )
        >>> tabular_df.head()
           image_path                         label            label_index    mask_path    split
        0  ./datasets/custom/images/00.png    DirType.NORMAL    0                           Split.TRAIN
        1  ./datasets/custom/images/01.png    DirType.NORMAL    0                           Split.TRAIN
        2  ./datasets/custom/images/02.png    DirType.NORMAL    0                           Split.TRAIN
        3  ./datasets/custom/images/03.png    DirType.NORMAL    0                           Split.TRAIN
        4  ./datasets/custom/images/04.png    DirType.NORMAL    0                           Split.TRAIN
    """
    ######################
    ### Pre-processing ###
    ######################

    # Convert to polars DataFrame if dictionary or list is given
    if isinstance(samples, AnomalibDataFrame):
        samples = samples.df
    if isinstance(samples, dict | list):
        samples = pl.DataFrame(samples)
    if not isinstance(samples, pl.DataFrame):
        msg = f"samples must be a dict, list, or polars DataFrame, found {type(samples)}"
        raise TypeError(msg)
    if "image_path" not in samples.columns:
        msg = "The samples table must contain an 'image_path' column."
        raise ValueError(msg)
    samples = samples.sort("image_path")

    ###########################
    ### Add missing columns ###
    ###########################

    samples = _infer_missing_columns(samples)

    #######################
    ### Post-processing ###
    #######################

    # Add root to paths
    samples = samples.with_columns(pl.col("mask_path").fill_null(""))
    if root:
        root_str = str(root)
        samples = samples.with_columns(
            pl.col("image_path").map_elements(lambda x: str(Path(root_str, x)), return_dtype=pl.Utf8),
        )
        samples = samples.with_columns(
            pl.when(pl.col("mask_path") != "")
            .then(pl.col("mask_path").map_elements(lambda x: str(Path(root_str, x)), return_dtype=pl.Utf8))
            .otherwise(pl.col("mask_path"))
            .alias("mask_path"),
        )
    samples = samples.with_columns(
        pl.col("image_path").cast(pl.Utf8),
        pl.col("mask_path").cast(pl.Utf8),
        pl.col("label").cast(pl.Utf8),
    )

    # Check if anomalous samples are in training set
    if (
        samples.filter(
            (pl.col("label_index") == int(LabelName.ABNORMAL)) & (pl.col("split") == Split.TRAIN.value),
        ).height
        > 0
    ):
        msg = "Training set must not contain anomalous samples."
        raise MisMatchError(msg)

    # Check for None or NaN values
    if samples.null_count().sum_horizontal().item() > 0:
        msg = "The samples table contains None or NaN values."
        raise ValueError(msg)

    # Infer the task type
    task = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    # Get the dataframe for the split.
    if split:
        samples = samples.filter(pl.col("split") == split)

    return AnomalibDataFrame(samples, attrs={"task": task})
