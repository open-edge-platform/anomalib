# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""BTech Dataset.

This module provides PyTorch Dataset implementation for the BTech dataset. The
dataset will be downloaded and extracted automatically if not found locally.

The dataset contains 3 categories of industrial objects with both normal and
anomalous samples. Each category includes RGB images and pixel-level ground truth
masks for anomaly segmentation.

License:
    BTech dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Mishra, P., Verk, C., Fornasier, D., & Piciarelli, C. (2021). VT-ADL: A
    Vision Transformer Network for Image Anomaly Detection and Localization. In
    IEEE International Conference on Image Processing (ICIP), 2021.
"""

from pathlib import Path

import polars as pl
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.utils import LabelName, Split, validate_path
from anomalib.data.utils.dataframe import AnomalibDataFrame

CATEGORIES = ("01", "02", "03")


class BTechDataset(AnomalibDataset):
    """BTech dataset class.

    Dataset class for loading and processing BTech dataset images. Supports both
    classification and segmentation tasks.

    Args:
        root (Path | str): Path to root directory containing the dataset.
        category (str): Category name, must be one of ``CATEGORIES``.
        transform (Transform | None, optional): Transforms to apply to the images.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import BTechDataset
        >>> dataset = BTechDataset(
        ...     root=Path("./datasets/btech"),
        ...     category="01",
        ...     split="train"
        ... )
        >>> dataset[0].keys()
        dict_keys(['image'])

        >>> dataset.split = "test"
        >>> dataset[0].keys()
        dict_keys(['image', 'image_path', 'label'])

        >>> # For segmentation task
        >>> dataset.split = "test"
        >>> dataset[0].keys()
        dict_keys(['image_path', 'label', 'mask_path', 'image', 'mask'])
        >>> dataset[0]["image"].shape, dataset[0]["mask"].shape
        (torch.Size([3, 256, 256]), torch.Size([256, 256]))
    """

    def __init__(
        self,
        root: str | Path,
        category: str,
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.root_category = Path(root) / category
        self.split = split
        self.samples = make_btech_dataset(path=self.root_category, split=self.split)


def make_btech_dataset(path: Path, split: str | Split | None = None) -> AnomalibDataFrame:
    """Create BTech samples by parsing the BTech data file structure.

    The files are expected to follow the structure:

    .. code-block:: bash

        path/to/dataset/
        ├── split/
        │   └── category/
        │       └── image_filename.png
        └── ground_truth/
            └── category/
                └── mask_filename.png

    Args:
        path (Path): Path to dataset directory.
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> path = Path("./datasets/btech/01")
        >>> samples = make_btech_dataset(path, split="train")
        >>> samples.head()
           path        split label image_path              mask_path          label_index
        0  BTech/01   train ok    BTech/01/train/ok/105.bmp BTech/01/gt/ok/105.png  0
        1  BTech/01   train ok    BTech/01/train/ok/017.bmp BTech/01/gt/ok/017.png  0

    Returns:
        DataFrame: DataFrame containing samples for the requested split.

    Raises:
        RuntimeError: If no images are found in the dataset directory.
    """
    path = validate_path(path)

    samples_list = [
        (str(path), *filename.parts[-3:]) for filename in path.glob("**/*") if filename.suffix in {".bmp", ".png"}
    ]
    if not samples_list:
        msg = f"Found 0 images in {path}"
        raise RuntimeError(msg)

    samples = pl.DataFrame(samples_list, schema=["path", "split", "label", "image_path"], orient="row")
    samples = samples.filter(pl.col("split") != "ground_truth")

    # Create mask_path column
    # (safely handles cases where non-mask image_paths end with either .png or .bmp)
    # Strip extension from image filename for mask path construction
    samples = samples.with_columns(
        (
            pl.col("path")
            + "/ground_truth/"
            + pl.col("label")
            + "/"
            + pl.col("image_path").str.replace(r"\.(png|bmp)$", "")
            + ".png"
        ).alias("mask_path"),
    )

    # Modify image_path column by converting to absolute path
    samples = samples.with_columns(
        (pl.col("path") + "/" + pl.col("split") + "/" + pl.col("label") + "/" + pl.col("image_path")).alias(
            "image_path",
        ),
    )

    # Good images don't have mask
    samples = samples.with_columns(
        pl.when((pl.col("split") == "test") & (pl.col("label") == "ok"))
        .then(pl.lit(""))
        .otherwise(pl.col("mask_path"))
        .alias("mask_path"),
    )

    # Create label index for normal (0) and anomalous (1) images.
    samples = samples.with_columns(
        pl.when(pl.col("label") == "ok")
        .then(pl.lit(int(LabelName.NORMAL)))
        .otherwise(pl.lit(int(LabelName.ABNORMAL)))
        .alias("label_index"),
    )

    # infer the task type
    task = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    if split:
        samples = samples.filter(pl.col("split") == split)

    return AnomalibDataFrame(samples, attrs={"task": task})
