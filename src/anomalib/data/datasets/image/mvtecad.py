# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MVTec AD Dataset.

This module provides PyTorch Dataset implementation for the MVTec AD dataset. The
dataset will be downloaded and extracted automatically if not found locally.

The dataset contains 15 categories of industrial objects with both normal and
anomalous samples. Each category includes RGB images and pixel-level ground truth
masks for anomaly segmentation.

License:
    MVTec AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., & Steger, C. (2021).
    The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for
    Unsupervised Anomaly Detection. International Journal of Computer Vision,
    129(4), 1038-1059.

    Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C. (2019). MVTec AD —
    A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. In
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
    9584-9592.
"""

from collections.abc import Sequence
from pathlib import Path

import polars as pl
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path
from anomalib.data.utils.dataframe import AnomalibDataFrame
from anomalib.utils import deprecate

IMG_EXTENSIONS = (".png", ".PNG")
CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


class MVTecADDataset(AnomalibDataset):
    """MVTec AD dataset class.

    Dataset class for loading and processing MVTec AD dataset images. Supports
    both classification and segmentation tasks.

    Args:
        root (Path | str): Path to root directory containing the dataset.
            Defaults to ``"./datasets/MVTecAD"``.
        category (str): Category name, must be one of ``CATEGORIES``.
            Defaults to ``"bottle"``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import MVTecADDataset
        >>> dataset = MVTecADDataset(
        ...     root=Path("./datasets/MVTecAD"),
        ...     category="bottle",
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
        root: Path | str = "./datasets/MVTecAD",
        category: str = "bottle",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.root_category = Path(root) / Path(category)
        self.category = category
        self.split = split
        self.samples = make_mvtec_ad_dataset(
            self.root_category,
            split=self.split,
            extensions=IMG_EXTENSIONS,
        )


def make_mvtec_ad_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
) -> AnomalibDataFrame:
    """Create MVTec AD samples by parsing the data directory structure.

    The files are expected to follow the structure:
        ``path/to/dataset/split/category/image_filename.png``
        ``path/to/dataset/ground_truth/category/mask_filename.png``

    Args:
        root (Path | str): Path to dataset root directory
        split (str | Split | None, optional): Dataset split (train or test)
            Defaults to ``None``.
        extensions (Sequence[str] | None, optional): Valid file extensions
            Defaults to ``None``.

    Returns:
        AnomalibDataFrame: Dataset samples with columns:
            - path: Base path to dataset
            - split: Dataset split (train/test)
            - label: Class label
            - image_path: Path to image file
            - mask_path: Path to mask file (if available)
            - label_index: Numeric label (0=normal, 1=abnormal)

    Raises:
        RuntimeError: If no valid images are found
        MisMatchError: If anomalous images and masks don't match
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)
    samples_list = [(str(root), *f.parts[-3:]) for f in root.glob(r"**/*") if f.suffix in extensions]
    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = pl.DataFrame(samples_list, schema=["path", "split", "label", "image_path"], orient="row")

    # Modify image_path column by converting to absolute path
    samples = samples.with_columns(
        (pl.col("path") + "/" + pl.col("split") + "/" + pl.col("label") + "/" + pl.col("image_path")).alias(
            "image_path",
        ),
    )

    # Create label index for normal (0) and anomalous (1) images.
    samples = samples.with_columns(
        pl.when(pl.col("label") == "good")
        .then(pl.lit(int(LabelName.NORMAL)))
        .otherwise(pl.lit(int(LabelName.ABNORMAL)))
        .alias("label_index"),
    )

    # separate masks from samples
    mask_samples = samples.filter(pl.col("split") == "ground_truth").sort("image_path")
    samples = samples.filter(pl.col("split") != "ground_truth").sort("image_path")

    # assign mask paths to anomalous test images
    samples = samples.with_columns(pl.lit("").alias("mask_path"))
    abnormal_test = samples.with_row_index("_idx").filter(
        (pl.col("split") == "test") & (pl.col("label_index") == int(LabelName.ABNORMAL)),
    )
    if len(abnormal_test) > 0 and len(mask_samples) > 0:
        update = pl.DataFrame(
            {"_idx": abnormal_test["_idx"], "_mask_path": mask_samples["image_path"][: len(abnormal_test)]},
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

    # assert that the right mask files are associated with the right test images
    abnormal_samples = samples.filter(pl.col("label_index") == int(LabelName.ABNORMAL))
    if len(abnormal_samples) > 0:
        for row in abnormal_samples.iter_rows(named=True):
            if row["mask_path"] and Path(row["image_path"]).stem not in Path(row["mask_path"]).stem:
                msg = (
                    "Mismatch between anomalous images and ground truth masks. Make sure "
                    "mask files in 'ground_truth' folder follow the same naming "
                    "convention as the anomalous images (e.g. image: '000.png', "
                    "mask: '000.png' or '000_mask.png')."
                )
                raise MisMatchError(msg)

    # infer the task type
    task = "classification" if (samples["mask_path"] == "").all() else "segmentation"

    if split:
        samples = samples.filter(pl.col("split") == split)

    return AnomalibDataFrame(samples, attrs={"task": task})


@deprecate(since="2.1.0", remove="2.3.0", use="MVTecADDataset")
class MVTecDataset(MVTecADDataset):
    """MVTec dataset class (Deprecated).

    This class is deprecated and will be removed in a future version.
    Please use MVTecADDataset instead.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
