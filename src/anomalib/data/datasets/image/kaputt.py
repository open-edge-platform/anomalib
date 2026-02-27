# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Kaputt Dataset.

This module provides PyTorch Dataset implementation for the Kaputt dataset.

The Kaputt dataset is a large-scale dataset for visual defect detection in
logistics settings. With over 230,000 images (and more than 29,000 defective
instances), it is 40 times larger than MVTec AD and contains more than 48,000
distinct objects.

The dataset uses Parquet files for metadata and has the following structure:
    datasets/
    ├── query-train.parquet
    ├── query-validation.parquet
    ├── query-test.parquet
    ├── reference-train.parquet
    ├── reference-validation.parquet
    └── reference-test.parquet

    query-image/data/<split>/query-data/image/      # Query images (may have defects)
    query-crop/data/<split>/query-data/crop/        # Cropped item regions
    query-mask/data/<split>/query-data/mask/        # Binary segmentation masks

    reference-image/data/<split>/reference-data/image/   # Reference (defect-free) images
    reference-crop/data/<split>/reference-data/crop/
    reference-mask/data/<split>/reference-data/mask/

License:
    The Kaputt dataset is released under the Creative Commons
    Attribution-NonCommercial-NoDerivatives 4.0 International License
    (CC BY-NC-ND 4.0) https://creativecommons.org/licenses/by-nc-nd/4.0/

Reference:
    Höfer, S., Henning, D., Amiranashvili, A., Morrison, D., Tzes, M.,
    Posner, I., Matvienko, M., Rennola, A., & Milan, A. (2025).
    Kaputt: A Large-Scale Dataset for Visual Defect Detection.
    In IEEE/CVF International Conference on Computer Vision (ICCV).

Dataset URL:
    https://www.kaputt-dataset.com/
"""

import logging
from pathlib import Path

import polars as pl
from lightning_utilities.core.imports import module_available
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.utils import LabelName, Split, validate_path
from anomalib.data.utils.dataframe import AnomalibDataFrame

logger = logging.getLogger(__name__)

# Material categories in Kaputt dataset (based on item_material field)
CATEGORIES = (
    "book_other",
    "book_paper",
    "book_plastic_tight_wrap",
    "cardboard",
    "other",
    "paper",
    "plastic_bubble_wrap",
    "plastic_hard",
    "plastic_loose_bag",
    "plastic_tight_wrap",
)


class KaputtDataset(AnomalibDataset):
    """Kaputt dataset class.

    Dataset class for loading and processing Kaputt dataset images. Supports
    both classification and segmentation tasks.

    The Kaputt dataset uses Parquet files for metadata instead of a directory
    structure. Images are organized as query (potentially defective) and
    reference (defect-free) sets.

    Args:
        root (Path | str): Path to root directory containing the dataset.
            Defaults to ``"./datasets/kaputt"``.
        augmentations (Transform, optional): Augmentations that should be applied
            to the input images. Defaults to ``None``.
        split (str | Split | None, optional): Dataset split - ``Split.TRAIN``,
            ``Split.VAL``, or ``Split.TEST``. Defaults to ``None``.
        image_type (str): Type of images to use - "image" for full images or
            "crop" for cropped item regions. Defaults to ``"image"``.
        use_reference (bool): If True, include reference (defect-free) images
            in addition to query images. Reference images are always labeled
            as normal. Defaults to ``False``.

    Example:
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import KaputtDataset
        >>> dataset = KaputtDataset(
        ...     root=Path("./datasets/kaputt"),
        ...     split="train"
        ... )

        For classification tasks, each sample contains:

        >>> sample = dataset[0]
        >>> list(sample.keys())
        ['image_path', 'label', 'image']

        For segmentation tasks, samples also include mask paths and masks:

        >>> dataset.task = "segmentation"
        >>> sample = dataset[0]
        >>> list(sample.keys())
        ['image_path', 'label', 'image', 'mask_path', 'mask']

        Images are PyTorch tensors with shape ``(C, H, W)``, masks have shape
        ``(H, W)``:

        >>> sample["image"].shape, sample["mask"].shape
        (torch.Size([3, 256, 256]), torch.Size([256, 256]))
    """

    def __init__(
        self,
        root: Path | str = "./datasets/kaputt",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
        image_type: str = "image",
        use_reference: bool = False,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.root = Path(root)
        self.split = split
        self.image_type = image_type
        self.use_reference = use_reference
        self.samples = make_kaputt_dataset(
            self.root,
            split=self.split,
            image_type=self.image_type,
            use_reference=self.use_reference,
        )


def make_kaputt_dataset(
    root: str | Path,
    split: str | Split | None = None,
    image_type: str = "image",
    use_reference: bool = False,
) -> AnomalibDataFrame:
    """Create Kaputt samples by parsing the Parquet metadata files.

    The Kaputt dataset uses Parquet files containing metadata about each image,
    including relative paths to image, crop, and mask files. Query parquets
    contain columns such as ``capture_id``, ``defect``, ``item_material``,
    ``query_image``, ``query_crop``, and ``query_mask``. Reference parquets
    contain ``item_identifier``, ``reference_image``, ``reference_crop``, and
    ``reference_mask``.

    Args:
        root (Path | str): Path to dataset root directory.
        split (str | Split | None, optional): Dataset split (train, val, or test).
            Defaults to ``None`` which loads all splits.
        image_type (str): Type of images - "image" for full images or "crop"
            for cropped regions. Defaults to ``"image"``.
        use_reference (bool): If True, include reference images which are
            always normal. Defaults to ``False``.

    Returns:
        DataFrame: Dataset samples with columns:
            - split: Dataset split (train/val/test)
            - label: Class label ("normal" or "abnormal")
            - image_path: Path to image file
            - mask_path: Path to mask file (for abnormal images)
            - label_index: Numeric label (0=normal, 1=abnormal)
            - capture_id: Unique capture identifier
            - defect_types: List of defect types (for abnormal images)
            - item_material: Material category

    Example:
        >>> root = Path("./datasets/kaputt")
        >>> samples = make_kaputt_dataset(root, split="train")
        >>> samples.head()
           split label image_path           mask_path label_index capture_id
        0  train normal [...]/image/abc.jpg                    0      abc

    Raises:
        FileNotFoundError: If the Parquet metadata files are not found.
        RuntimeError: If no valid images are found.
    """
    root = validate_path(root)

    # Parquet files use "validation" while anomalib uses "val"
    parquet_splits = {"train": "train", "validation": "val", "test": "test"}

    frames: list[pl.DataFrame] = []

    if not module_available("pyarrow"):
        msg = (
            "pyarrow is needed to read the parquet files. You can install it using: `uv pip install pyarrow`"
            "or `uv pip install anomalib[datasets]`"
        )
        raise ImportError(msg)

    for parquet_name, anomalib_name in parquet_splits.items():
        # --- Query samples ---
        query_parquet = root / "datasets" / f"query-{parquet_name}.parquet"
        if not query_parquet.exists():
            msg = f"Query parquet file not found: {query_parquet}"
            raise FileNotFoundError(msg)

        query_df = pl.read_parquet(query_parquet)

        # Build image paths: root / "query-{image_type}" / <relative path from parquet>
        image_col = f"query_{image_type}"  # "query_image" or "query_crop"
        root_prefix = str(root / f"query-{image_type}") + "/"
        mask_prefix = str(root / "query-mask") + "/"

        samples = pl.DataFrame({
            "image_path": query_df[image_col].cast(pl.Utf8),
            "capture_id": query_df["capture_id"],
            "item_material": query_df["item_material"].fill_null(""),
            "defect_types": query_df["defect_types"].fill_null(""),
        }).with_columns(
            (pl.lit(root_prefix) + pl.col("image_path")).alias("image_path"),
            pl.lit(anomalib_name).alias("split"),
        )

        # Label assignment
        defect_col = query_df["defect"].fill_null(value=False).cast(pl.Boolean)
        samples = samples.with_columns(defect_col.alias("_defect"))
        samples = (
            samples.with_columns(
                pl.when(pl.col("_defect").not_())
                .then(pl.lit(int(LabelName.NORMAL)))
                .otherwise(pl.lit(int(LabelName.ABNORMAL)))
                .alias("label_index"),
            )
            .with_columns(
                pl.when(pl.col("label_index") == int(LabelName.NORMAL))
                .then(pl.lit("normal"))
                .otherwise(pl.lit("abnormal"))
                .alias("label"),
            )
            .drop("_defect")
        )

        # Mask paths: only for defective images
        samples = samples.with_columns(query_df["query_mask"].fill_null("").alias("_query_mask"))
        samples = samples.with_columns(
            pl.when(pl.col("label_index") == int(LabelName.ABNORMAL))
            .then(pl.lit(mask_prefix) + pl.col("_query_mask"))
            .otherwise(pl.lit(""))
            .alias("mask_path"),
        ).drop("_query_mask")

        frames.append(samples)

        # --- Reference samples (always normal) ---
        if use_reference:
            ref_parquet = root / "datasets" / f"reference-{parquet_name}.parquet"
            if ref_parquet.exists():
                ref_df = pl.read_parquet(ref_parquet)
                ref_image_col = f"reference_{image_type}"
                ref_prefix = str(root / f"reference-{image_type}") + "/"

                ref_samples = pl.DataFrame({
                    "image_path": ref_df[ref_image_col].cast(pl.Utf8),
                    "capture_id": ref_df["item_identifier"],
                }).with_columns(
                    (pl.lit(ref_prefix) + pl.col("image_path")).alias("image_path"),
                    pl.lit("").alias("item_material"),
                    pl.lit("").alias("defect_types"),
                    pl.lit(anomalib_name).alias("split"),
                    pl.lit(int(LabelName.NORMAL)).alias("label_index"),
                    pl.lit("normal").alias("label"),
                    pl.lit("").alias("mask_path"),
                )

                frames.append(ref_samples)
            else:
                msg = f"Reference parquet file not found: {ref_parquet}"
                logger.warning(msg)

    if not frames:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = pl.concat(frames).sort("image_path")

    # infer the task type
    task = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    if split:
        samples = samples.filter(pl.col("split") == split)

    return AnomalibDataFrame(samples, attrs={"task": task})
