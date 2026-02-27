# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""3D-ADAM Datamodule.

This module provides PyTorch Dataset for
the 3D-ADAM dataset. If the dataset is not available locally, it will be
downloaded and extracted automatically.

License:
    3D-ADAM dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)
    https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference: https://arxiv.org/abs/2507.07838

"""

from collections.abc import Sequence
from pathlib import Path

import polars as pl
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.depth import AnomalibDepthDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import AnomalibDataFrame, LabelName, Split, validate_path

IMG_EXTENSIONS = [".png", ".PNG", ".tiff"]
CATEGORIES = (
    "1m1",
    "1m2",
    "1m3",
    "2m1",
    "2m2h",
    "2m2m",
    "3m1",
    "3m2",
    "3m2c",
    "4m1",
    "4m2",
    "4m2c",
    "gripper_closed",
    "gripper_open",
    "helicalgear1",
    "helicalgear2",
    "rackgear",
    "spiralgear",
    "spurgear",
    "tapa2m1",
    "tapa3m1",
    "tapa4m1",
    "tapatbb",
)


class ADAM3DDataset(AnomalibDepthDataset):
    """3D ADAM dataset class.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/ADAM3D"``.
        category (str): Category name, e.g. ``"1m1"``.
            Defaults to ``"1m1"``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None): Dataset split - usually ``Split.TRAIN`` or
            ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> dataset = ADAM3DDataset(
        ...     root=Path("./datasets/ADAM3D"),
        ...     category="1m1",
        ...     split="train"
        ... )
    """

    def __init__(
        self,
        root: Path | str = "./datasets/Adam3D",
        category: str = "1m1",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.root_category = Path(root) / Path(category)
        self.split = split
        self.samples = make_adam_3d_dataset(
            self.root_category,
            split=self.split,
            extensions=IMG_EXTENSIONS,
        )


def make_adam_3d_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
) -> AnomalibDataFrame:
    """Create 3D-ADAM samples by parsing the data directory structure.

    The files are expected to follow this structure::

        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/mask_filename.png

    The function creates a DataFrame with the following format::

        +---+---------------+-------+---------+---------------+--------------------+
        |   | path          | split | label   | image_path    | mask_path         |
        +---+---------------+-------+---------+---------------+--------------------+
        | 0 | datasets/name | test  | defect  | filename.png  | defect/mask.png   |
        +---+---------------+-------+---------+---------------+--------------------+

    Args:
        root (Path | str): Path to the dataset root directory.
        split (str | Split | None, optional): Dataset split (e.g., ``"train"`` or
            ``"test"``). Defaults to ``None``.
        extensions (Sequence[str] | None, optional): List of valid file extensions.
            Defaults to ``None``.

    Returns:
        AnomalibDataFrame: DataFrame containing the dataset samples.

    Example:
        >>> from pathlib import Path
        >>> root = Path("./datasets/ADAM3D/1m1")
        >>> samples = make_adam_3d_dataset(root, split="train")
        >>> samples.head()
           path     split label image_path                  mask_path
        0  ADAM3D  train good  train/good/rgb/001_C.png     ground_truth/001_C.png
        1  ADAM3D  train good  train/good/rgb/015_D.png     ground_truth/015_D.png

    Raises:
        RuntimeError: If no images are found in the root directory.
        MisMatchError: If there is a mismatch between images and their
            corresponding mask/depth files.
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)
    samples_list = [(str(root), *f.parts[-4:]) for f in root.glob(r"**/*") if f.suffix in extensions]
    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = pl.DataFrame(
        samples_list,
        schema=["path", "split", "label", "type", "file_name"],
        orient="row",
    )

    # Modify image_path column by converting to absolute path
    samples = samples.with_columns(
        pl.when(pl.col("type") == "rgb")
        .then(pl.col("path") + "/" + pl.col("split") + "/" + pl.col("label") + "/" + "rgb/" + pl.col("file_name"))
        .otherwise(pl.lit(None))
        .alias("image_path"),
    )
    samples = samples.with_columns(
        pl.when(pl.col("type") == "rgb")
        .then(
            pl.col("path")
            + "/"
            + pl.col("split")
            + "/"
            + pl.col("label")
            + "/"
            + "xyz/"
            + pl.col("file_name").str.split(".").list.first()
            + ".tiff",
        )
        .otherwise(pl.lit(None))
        .alias("depth_path"),
    )

    # Create label index for normal (0) and anomalous (1) images.
    samples = samples.with_columns(
        pl.when(pl.col("label") == "good")
        .then(pl.lit(int(LabelName.NORMAL)))
        .otherwise(pl.lit(int(LabelName.ABNORMAL)))
        .alias("label_index"),
    )

    # separate masks from samples
    mask_samples = samples.filter(
        (pl.col("split") == "test") & (pl.col("type") == "rgb"),
    ).sort(by="image_path")
    samples = samples.sort(by="image_path")

    # assign mask paths to all test images
    if "mask_path" not in samples.columns:
        samples = samples.with_columns(pl.lit(None).cast(pl.Utf8).alias("mask_path"))

    test_rgb_indices = samples.with_row_index("_idx").filter(
        (pl.col("split") == "test") & (pl.col("type") == "rgb"),
    )["_idx"]

    if len(test_rgb_indices) > 0 and len(mask_samples) > 0:
        mask_paths = (
            mask_samples["path"]
            + "/"
            + mask_samples["split"]
            + "/"
            + mask_samples["label"]
            + "/"
            + "ground_truth/"
            + mask_samples["file_name"]
        )
        update = pl.DataFrame({"_idx": test_rgb_indices, "_mask_path": mask_paths})
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

    samples = samples.drop_nulls(subset=["image_path"])
    samples = samples.with_columns(
        pl.col("image_path").cast(pl.Utf8),
        pl.col("mask_path").fill_null("").cast(pl.Utf8),
        pl.col("depth_path").fill_null("").cast(pl.Utf8),
    )

    # assert that the right mask files are associated with the right test images
    mismatch_masks = all(
        Path(row["image_path"]).stem in Path(row["mask_path"]).stem
        for row in samples.filter(pl.col("label_index") == int(LabelName.ABNORMAL)).iter_rows(named=True)
    )
    if not mismatch_masks:
        msg = (
            "Mismatch between anomalous images and ground truth masks. Ensure mask "
            "files in 'ground_truth' folder follow the same naming convention as "
            "the anomalous images (e.g. image: '000.png', mask: '000.png')."
        )
        raise MisMatchError(msg)

    mismatch_depth = all(
        Path(row["image_path"]).stem in Path(row["depth_path"]).stem
        for row in samples.filter(pl.col("label_index") == int(LabelName.ABNORMAL)).iter_rows(named=True)
    )
    if not mismatch_depth:
        msg = (
            "Mismatch between anomalous images and depth images. Ensure depth "
            "files in 'xyz' folder follow the same naming convention as the "
            "anomalous images (e.g. image: '000.png', depth: '000.tiff')."
        )
        raise MisMatchError(msg)

    # infer the task type
    task = "classification" if samples.select((pl.col("mask_path") == "").all()).item() else "segmentation"

    if split:
        samples = samples.filter(pl.col("split") == split)

    return AnomalibDataFrame(samples, attrs={"task": task})
