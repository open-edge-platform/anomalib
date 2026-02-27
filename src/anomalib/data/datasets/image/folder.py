# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom Folder Dataset.

This module provides a custom PyTorch Dataset implementation for loading images
from a folder structure. The dataset supports both classification and
segmentation tasks.

The folder structure should contain normal images and optionally abnormal images,
test images, and mask annotations.

Example:
    >>> from pathlib import Path
    >>> from anomalib.data.datasets import FolderDataset
    >>> dataset = FolderDataset(
    ...     name="custom",
    ...     root="datasets/custom",
    ...     normal_dir="normal",
    ...     abnormal_dir="abnormal",
    ...     mask_dir="ground_truth"
    ... )
"""

from collections.abc import Sequence
from pathlib import Path

import polars as pl
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import DirType, LabelName, Split
from anomalib.data.utils.dataframe import AnomalibDataFrame
from anomalib.data.utils.path import _prepare_files_labels, validate_and_resolve_path


class FolderDataset(AnomalibDataset):
    """Dataset class for loading images from a custom folder structure.

    Args:
        name (str): Name of the dataset. Used for logging/saving.
        normal_dir (str | Path | Sequence): Path to directory containing normal
            images.
        transform (Transform | None, optional): Transforms to apply to the images.
            Defaults to ``None``.
        root (str | Path | None, optional): Root directory of the dataset.
            Defaults to ``None``.
        abnormal_dir (str | Path | Sequence | None, optional): Path to directory
            containing abnormal images. Defaults to ``None``.
        normal_test_dir (str | Path | Sequence | None, optional): Path to
            directory containing normal test images. If not provided, normal test
            images will be split from ``normal_dir``. Defaults to ``None``.
        mask_dir (str | Path | Sequence | None, optional): Path to directory
            containing ground truth masks. Required for segmentation.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split to load.
            Choose from ``Split.FULL``, ``Split.TRAIN``, ``Split.TEST``.
            Defaults to ``None``.
        extensions (tuple[str, ...] | None, optional): Image file extensions to
            include. Defaults to ``None``.

    Examples:
        Create a classification dataset:

        >>> from anomalib.data.utils import InputNormalizationMethod, get_transforms
        >>> transform = get_transforms(
        ...     image_size=256,
        ...     normalization=InputNormalizationMethod.NONE
        ... )
        >>> dataset = FolderDataset(
        ...     name="custom",
        ...     normal_dir="datasets/custom/good",
        ...     abnormal_dir="datasets/custom/defect",
        ...     split="train",
        ...     transform=transform
        ... )

        Create a segmentation dataset:

        >>> dataset = FolderDataset(
        ...     name="custom",
        ...     normal_dir="datasets/custom/good",
        ...     abnormal_dir="datasets/custom/defect",
        ...     mask_dir="datasets/custom/ground_truth",
        ...     split="test"
        ... )
    """

    def __init__(
        self,
        name: str,
        normal_dir: str | Path | Sequence[str | Path],
        augmentations: Transform | None = None,
        root: str | Path | None = None,
        abnormal_dir: str | Path | Sequence[str | Path] | None = None,
        normal_test_dir: str | Path | Sequence[str | Path] | None = None,
        mask_dir: str | Path | Sequence[str | Path] | None = None,
        split: str | Split | None = None,
        extensions: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self._name = name
        self.split = split
        self.root = root
        self.normal_dir = normal_dir
        self.abnormal_dir = abnormal_dir
        self.normal_test_dir = normal_test_dir
        self.mask_dir = mask_dir
        self.extensions = extensions

        self.samples = make_folder_dataset(
            root=self.root,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            mask_dir=self.mask_dir,
            split=self.split,
            extensions=self.extensions,
        )

    @property
    def name(self) -> str:
        """Get dataset name.

        Returns:
            str: Name of the dataset
        """
        return self._name


def make_folder_dataset(
    normal_dir: str | Path | Sequence[str | Path],
    root: str | Path | None = None,
    abnormal_dir: str | Path | Sequence[str | Path] | None = None,
    normal_test_dir: str | Path | Sequence[str | Path] | None = None,
    mask_dir: str | Path | Sequence[str | Path] | None = None,
    split: str | Split | None = None,
    extensions: tuple[str, ...] | None = None,
) -> AnomalibDataFrame:
    """Create a dataset from a folder structure.

    Args:
        normal_dir (str | Path | Sequence): Path to directory containing normal
            images.
        root (str | Path | None, optional): Root directory of the dataset.
            Defaults to ``None``.
        abnormal_dir (str | Path | Sequence | None, optional): Path to directory
            containing abnormal images. Defaults to ``None``.
        normal_test_dir (str | Path | Sequence | None, optional): Path to
            directory containing normal test images. If not provided, normal test
            images will be split from ``normal_dir``. Defaults to ``None``.
        mask_dir (str | Path | Sequence | None, optional): Path to directory
            containing ground truth masks. Required for segmentation.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split to load.
            Choose from ``Split.FULL``, ``Split.TRAIN``, ``Split.TEST``.
            Defaults to ``None``.
        extensions (tuple[str, ...] | None, optional): Image file extensions to
            include. Defaults to ``None``.

    Returns:
        DataFrame: Dataset samples with columns for image paths, labels, splits
            and mask paths (for segmentation).

    Examples:
        Create a classification dataset:

        >>> folder_df = make_folder_dataset(
        ...     normal_dir="datasets/custom/good",
        ...     abnormal_dir="datasets/custom/defect",
        ...     split="train"
        ... )
        >>> folder_df.head()
                  image_path           label  label_index mask_path    split
        0  ./good/00.png     DirType.NORMAL            0            Split.TRAIN
        1  ./good/01.png     DirType.NORMAL            0            Split.TRAIN
        2  ./good/02.png     DirType.NORMAL            0            Split.TRAIN
        3  ./good/03.png     DirType.NORMAL            0            Split.TRAIN
        4  ./good/04.png     DirType.NORMAL            0            Split.TRAIN
    """

    def _resolve_path_and_convert_to_list(path: str | Path | Sequence[str | Path] | None) -> list[Path]:
        """Convert path to list of paths.

        Args:
            path (str | Path | Sequence | None): Path to convert.

        Returns:
            list[Path]: List of resolved paths.

        Examples:
            >>> _resolve_path_and_convert_to_list("dir")
            [Path("path/to/dir")]
            >>> _resolve_path_and_convert_to_list(["dir1", "dir2"])
            [Path("path/to/dir1"), Path("path/to/dir2")]
        """
        if isinstance(path, Sequence) and not isinstance(path, str):
            return [validate_and_resolve_path(dir_path, root) for dir_path in path]
        return [validate_and_resolve_path(path, root)] if path is not None else []

    # All paths are changed to the List[Path] type and used.
    normal_dir = _resolve_path_and_convert_to_list(normal_dir)
    abnormal_dir = _resolve_path_and_convert_to_list(abnormal_dir)
    normal_test_dir = _resolve_path_and_convert_to_list(normal_test_dir)
    mask_dir = _resolve_path_and_convert_to_list(mask_dir)
    if len(normal_dir) == 0:
        msg = "A folder location must be provided in normal_dir."
        raise ValueError(msg)

    filenames = []
    labels = []
    dirs = {DirType.NORMAL: normal_dir}

    if abnormal_dir:
        dirs[DirType.ABNORMAL] = abnormal_dir

    if normal_test_dir:
        dirs[DirType.NORMAL_TEST] = normal_test_dir

    if mask_dir:
        dirs[DirType.MASK] = mask_dir

    for dir_type, paths in dirs.items():
        for path in paths:
            filename, label = _prepare_files_labels(path, dir_type, extensions)
            filenames += filename
            labels += label

    samples = pl.DataFrame({"image_path": [str(f) for f in filenames], "label": [str(lbl) for lbl in labels]})
    samples = samples.sort("image_path")

    # Create label index for normal (0) and abnormal (1) images.
    samples = samples.with_columns(
        pl.when((pl.col("label") == str(DirType.NORMAL)) | (pl.col("label") == str(DirType.NORMAL_TEST)))
        .then(pl.lit(int(LabelName.NORMAL)))
        .when(pl.col("label") == str(DirType.ABNORMAL))
        .then(pl.lit(int(LabelName.ABNORMAL)))
        .otherwise(pl.lit(None).cast(pl.Int64))
        .alias("label_index"),
    )

    # If a path to mask is provided, add it to the sample dataframe.

    if len(mask_dir) > 0 and len(abnormal_dir) > 0:
        mask_paths = samples.filter(pl.col("label") == str(DirType.MASK))["image_path"]
        abnormal_rows = samples.with_row_index("_idx").filter(pl.col("label") == str(DirType.ABNORMAL))

        samples = samples.with_columns(pl.lit("").alias("mask_path"))
        if len(abnormal_rows) > 0 and len(mask_paths) > 0:
            update = pl.DataFrame(
                {"_idx": abnormal_rows["_idx"], "_mask_path": mask_paths[: len(abnormal_rows)]},
            )
            samples = (
                samples.with_row_index("_idx")
                .join(update, on="_idx", how="left")
                .with_columns(
                    pl.when(pl.col("_mask_path").is_not_null())
                    .then(pl.col("_mask_path"))
                    .otherwise(pl.col("mask_path"))
                    .alias("mask_path"),
                )
                .drop("_idx", "_mask_path")
            )

        # make sure every rgb image has a corresponding mask image.
        abnormal_samples = samples.filter(pl.col("label_index") == int(LabelName.ABNORMAL))
        for row in abnormal_samples.iter_rows(named=True):
            if row["mask_path"] and Path(row["image_path"]).stem not in Path(row["mask_path"]).stem:
                msg = """Mismatch between anomalous images and mask images. Make sure
                    the mask files folder follow the same naming convention as the
                    anomalous images in the dataset (e.g. image: '000.png',
                    mask: '000.png')."""
                raise MisMatchError(msg)

    else:
        samples = samples.with_columns(pl.lit("").alias("mask_path"))

    # remove all the rows with temporal image samples that have already been
    # assigned
    samples = samples.filter(
        (pl.col("label") == str(DirType.NORMAL))
        | (pl.col("label") == str(DirType.ABNORMAL))
        | (pl.col("label") == str(DirType.NORMAL_TEST)),
    )

    # Ensure the pathlib objects are converted to str.
    # This is because torch dataloader doesn't like pathlib.
    samples = samples.with_columns(pl.col("image_path").cast(pl.Utf8))

    # Create train/test split.
    # By default, all the normal samples are assigned as train.
    #   and all the abnormal samples are test.
    samples = samples.with_columns(
        pl.when(pl.col("label") == str(DirType.NORMAL))
        .then(pl.lit(Split.TRAIN))
        .otherwise(pl.lit(Split.TEST))
        .alias("split"),
    )

    # infer the task type
    task = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    # Get the data frame for the split.
    if split:
        samples = samples.filter(pl.col("split") == split)

    return AnomalibDataFrame(samples, attrs={"task": task})
