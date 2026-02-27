# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom Folder Dataset for 3D anomaly detection.

This module provides a custom dataset class that loads RGB-D data from a folder
structure. The dataset supports both classification and segmentation tasks.

The folder structure should contain RGB images and their corresponding depth maps.
The dataset can be configured with separate directories for:

- Normal training samples
- Normal test samples (optional)
- Abnormal test samples (optional)
- Mask annotations (optional, for segmentation)
- Depth maps for each image type

Example:
    >>> from pathlib import Path
    >>> from anomalib.data.datasets import Folder3DDataset
    >>> dataset = Folder3DDataset(
    ...     name="custom",
    ...     root="datasets/custom",
    ...     normal_dir="normal",
    ...     abnormal_dir="abnormal",
    ...     normal_depth_dir="normal_depth",
    ...     abnormal_depth_dir="abnormal_depth",
    ...     mask_dir="ground_truth"
    ... )
"""

from pathlib import Path

import polars as pl
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.depth import AnomalibDepthDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import AnomalibDataFrame, DirType, LabelName, Split
from anomalib.data.utils.path import _prepare_files_labels, validate_and_resolve_path


class Folder3DDataset(AnomalibDepthDataset):
    """Dataset class for loading RGB-D data from a custom folder structure.

    Args:
        name (str): Name of the dataset
        normal_dir (str | Path): Path to directory containing normal images
        root (str | Path | None, optional): Root directory of the dataset.
            Defaults to ``None``.
        abnormal_dir (str | Path | None, optional): Path to directory containing
            abnormal images. Defaults to ``None``.
        normal_test_dir (str | Path | None, optional): Path to directory
            containing normal test images. If not provided, normal test images
            will be split from ``normal_dir``. Defaults to ``None``.
        mask_dir (str | Path | None, optional): Path to directory containing
            ground truth masks. Required for segmentation. Defaults to ``None``.
        normal_depth_dir (str | Path | None, optional): Path to directory
            containing depth maps for normal images. Defaults to ``None``.
        abnormal_depth_dir (str | Path | None, optional): Path to directory
            containing depth maps for abnormal images. Defaults to ``None``.
        normal_test_depth_dir (str | Path | None, optional): Path to directory
            containing depth maps for normal test images. Defaults to ``None``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split to load.
            One of ``["train", "test", "full"]``. Defaults to ``None``.
        extensions (tuple[str, ...] | None, optional): Image file extensions to
            include. Defaults to ``None``.

    Example:
        >>> dataset = Folder3DDataset(
        ...     name="custom",
        ...     root="./datasets/custom",
        ...     normal_dir="train/good",
        ...     abnormal_dir="test/defect",
        ...     mask_dir="test/ground_truth",
        ...     normal_depth_dir="train/good_depth",
        ...     abnormal_depth_dir="test/defect_depth"
        ... )
    """

    def __init__(
        self,
        name: str,
        normal_dir: str | Path,
        root: str | Path | None = None,
        abnormal_dir: str | Path | None = None,
        normal_test_dir: str | Path | None = None,
        mask_dir: str | Path | None = None,
        normal_depth_dir: str | Path | None = None,
        abnormal_depth_dir: str | Path | None = None,
        normal_test_depth_dir: str | Path | None = None,
        augmentations: Transform | None = None,
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
        self.normal_depth_dir = normal_depth_dir
        self.abnormal_depth_dir = abnormal_depth_dir
        self.normal_test_depth_dir = normal_test_depth_dir
        self.extensions = extensions

        self.samples = make_folder3d_dataset(
            root=self.root,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            mask_dir=self.mask_dir,
            normal_depth_dir=self.normal_depth_dir,
            abnormal_depth_dir=self.abnormal_depth_dir,
            normal_test_depth_dir=self.normal_test_depth_dir,
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


def make_folder3d_dataset(  # noqa: C901
    normal_dir: str | Path,
    root: str | Path | None = None,
    abnormal_dir: str | Path | None = None,
    normal_test_dir: str | Path | None = None,
    mask_dir: str | Path | None = None,
    normal_depth_dir: str | Path | None = None,
    abnormal_depth_dir: str | Path | None = None,
    normal_test_depth_dir: str | Path | None = None,
    split: str | Split | None = None,
    extensions: tuple[str, ...] | None = None,
) -> AnomalibDataFrame:
    """Create a dataset by collecting files from a folder structure.

    The function creates a DataFrame containing paths to RGB images, depth maps,
    and masks (if available) along with their corresponding labels.

    Args:
        normal_dir (str | Path): Directory containing normal images
        root (str | Path | None, optional): Root directory. Defaults to ``None``.
        abnormal_dir (str | Path | None, optional): Directory containing abnormal
            images. Defaults to ``None``.
        normal_test_dir (str | Path | None, optional): Directory containing
            normal test images. Defaults to ``None``.
        mask_dir (str | Path | None, optional): Directory containing ground truth
            masks. Defaults to ``None``.
        normal_depth_dir (str | Path | None, optional): Directory containing
            depth maps for normal images. Defaults to ``None``.
        abnormal_depth_dir (str | Path | None, optional): Directory containing
            depth maps for abnormal images. Defaults to ``None``.
        normal_test_depth_dir (str | Path | None, optional): Directory containing
            depth maps for normal test images. Defaults to ``None``.
        split (str | Split | None, optional): Dataset split to return.
            Defaults to ``None``.
        extensions (tuple[str, ...] | None, optional): Image file extensions to
            include. Defaults to ``None``.

    Returns:
        DataFrame: Dataset samples with columns for paths and labels

    Raises:
        ValueError: If ``normal_dir`` is not a directory
        FileNotFoundError: If depth maps or mask files are missing
        MisMatchError: If depth maps don't match their RGB images
    """
    normal_dir = validate_and_resolve_path(normal_dir, root)
    abnormal_dir = validate_and_resolve_path(abnormal_dir, root) if abnormal_dir else None
    normal_test_dir = validate_and_resolve_path(normal_test_dir, root) if normal_test_dir else None
    mask_dir = validate_and_resolve_path(mask_dir, root) if mask_dir else None
    normal_depth_dir = validate_and_resolve_path(normal_depth_dir, root) if normal_depth_dir else None
    abnormal_depth_dir = validate_and_resolve_path(abnormal_depth_dir, root) if abnormal_depth_dir else None
    normal_test_depth_dir = validate_and_resolve_path(normal_test_depth_dir, root) if normal_test_depth_dir else None

    if not normal_dir.is_dir():
        msg = "A folder location must be provided in normal_dir."
        raise ValueError(msg)

    dirs = {
        DirType.NORMAL: normal_dir,
        DirType.ABNORMAL: abnormal_dir,
        DirType.NORMAL_TEST: normal_test_dir,
        DirType.NORMAL_DEPTH: normal_depth_dir,
        DirType.ABNORMAL_DEPTH: abnormal_depth_dir,
        DirType.NORMAL_TEST_DEPTH: normal_test_depth_dir,
        DirType.MASK: mask_dir,
    }

    filenames: list[Path] = []
    labels: list[str] = []

    for dir_type, dir_path in dirs.items():
        if dir_path is not None:
            filename, label = _prepare_files_labels(dir_path, dir_type, extensions)
            filenames += filename
            labels += label

    samples = pl.DataFrame({"image_path": [str(f) for f in filenames], "label": labels})
    samples = samples.sort(by="image_path")

    samples = samples.with_columns(
        pl.when((pl.col("label") == DirType.NORMAL) | (pl.col("label") == DirType.NORMAL_TEST))
        .then(pl.lit(int(LabelName.NORMAL)))
        .when(pl.col("label") == DirType.ABNORMAL)
        .then(pl.lit(int(LabelName.ABNORMAL)))
        .otherwise(pl.lit(None))
        .alias("label_index"),
    )

    if normal_depth_dir:
        # Assign depth paths using row-index join for normal images
        normal_indices = samples.with_row_index("_idx").filter(pl.col("label") == DirType.NORMAL)["_idx"]
        normal_depth_paths = samples.filter(pl.col("label") == DirType.NORMAL_DEPTH)["image_path"]

        abnormal_indices = samples.with_row_index("_idx").filter(pl.col("label") == DirType.ABNORMAL)["_idx"]
        abnormal_depth_paths = samples.filter(pl.col("label") == DirType.ABNORMAL_DEPTH)["image_path"]

        updates = []
        if len(normal_indices) > 0 and len(normal_depth_paths) > 0:
            updates.append(pl.DataFrame({"_idx": normal_indices, "_depth_path": normal_depth_paths}))
        if len(abnormal_indices) > 0 and len(abnormal_depth_paths) > 0:
            updates.append(pl.DataFrame({"_idx": abnormal_indices, "_depth_path": abnormal_depth_paths}))

        if normal_test_dir:
            normal_test_indices = samples.with_row_index("_idx").filter(
                pl.col("label") == DirType.NORMAL_TEST,
            )["_idx"]
            normal_test_depth_paths = samples.filter(
                pl.col("label") == DirType.NORMAL_TEST_DEPTH,
            )["image_path"]
            if len(normal_test_indices) > 0 and len(normal_test_depth_paths) > 0:
                updates.append(
                    pl.DataFrame({"_idx": normal_test_indices, "_depth_path": normal_test_depth_paths}),
                )

        samples = samples.with_columns(pl.lit(None).cast(pl.Utf8).alias("depth_path"))
        if updates:
            update_df = pl.concat(updates)
            samples = (
                samples.with_row_index("_idx")
                .join(update_df, on="_idx", how="left")
                .with_columns(
                    pl.when(pl.col("_depth_path").is_not_null())
                    .then(pl.col("_depth_path"))
                    .otherwise(pl.col("depth_path"))
                    .alias("depth_path"),
                )
                .drop("_idx", "_depth_path")
            )

        # make sure every rgb image has a corresponding depth image and that the file exists
        mismatch = all(
            Path(row["image_path"]).stem in Path(row["depth_path"]).stem
            for row in samples.filter(pl.col("label_index") == int(LabelName.ABNORMAL)).iter_rows(named=True)
            if row["depth_path"] is not None
        )
        if not mismatch:
            msg = (
                "Mismatch between anomalous images and depth images. "
                "Make sure the mask files in 'xyz' folder follow the same naming "
                "convention as the anomalous images in the dataset"
                "(e.g. image: '000.png', depth: '000.tiff')."
            )
            raise MisMatchError(msg)

        missing_depth_files = all(
            Path(row["depth_path"]).exists() if row["depth_path"] is not None else True
            for row in samples.iter_rows(named=True)
        )
        if not missing_depth_files:
            msg = "Missing depth image files."
            raise FileNotFoundError(msg)

        samples = samples.with_columns(pl.col("depth_path").fill_null("").cast(pl.Utf8))

    # If a path to mask is provided, add it to the sample dataframe.
    if mask_dir and abnormal_dir:
        abnormal_mask_indices = samples.with_row_index("_idx").filter(
            pl.col("label") == DirType.ABNORMAL,
        )["_idx"]
        mask_paths = samples.filter(pl.col("label") == DirType.MASK)["image_path"]

        samples = samples.with_columns(pl.lit("").alias("mask_path"))
        if len(abnormal_mask_indices) > 0 and len(mask_paths) > 0:
            update = pl.DataFrame({"_idx": abnormal_mask_indices, "_mask_path": mask_paths})
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

        # Make sure all the files exist
        if not all(
            Path(row["mask_path"]).exists() if row["mask_path"] != "" else True for row in samples.iter_rows(named=True)
        ):
            msg = f"Missing mask files. mask_dir={mask_dir}"
            raise FileNotFoundError(msg)
    else:
        samples = samples.with_columns(pl.lit("").alias("mask_path"))

    # Remove all the rows with temporal image samples that have already been assigned
    samples = samples.filter(
        (pl.col("label") == DirType.NORMAL)
        | (pl.col("label") == DirType.ABNORMAL)
        | (pl.col("label") == DirType.NORMAL_TEST),
    )

    # Ensure the pathlib objects are converted to str.
    # This is because torch dataloader doesn't like pathlib.
    samples = samples.with_columns(pl.col("image_path").cast(pl.Utf8))

    # Create train/test split.
    # By default, all the normal samples are assigned as train.
    #   and all the abnormal samples are test.
    samples = samples.with_columns(
        pl.when(pl.col("label") == DirType.NORMAL)
        .then(pl.lit(Split.TRAIN))
        .when((pl.col("label") == DirType.ABNORMAL) | (pl.col("label") == DirType.NORMAL_TEST))
        .then(pl.lit(Split.TEST))
        .otherwise(pl.lit(None))
        .alias("split"),
    )

    # infer the task type
    task = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    # Get the data frame for the split.
    if split:
        samples = samples.filter(pl.col("split") == split)

    return AnomalibDataFrame(samples, attrs={"task": task})
