# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MVTec LOCO AD Dataset (CC BY-NC-SA 4.0).

Description:
    This script contains PyTorch Dataset, Dataloader and PyTorch Lightning
    DataModule for the MVTec LOCO AD dataset. If the dataset is not on the file system,
    the script downloads and extracts the dataset and create PyTorch data objects.

License:
    MVTec LOCO AD dataset is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0)(https://creativecommons.org/licenses/by-nc-sa/4.0/).

References:
    - Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger:
      Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization;
      in: International Journal of Computer Vision (IJCV) 130, 947-969, 2022, DOI: 10.1007/s11263-022-01578-9
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import polars as pl
import torch
from torchvision.transforms.v2 import Transform
from torchvision.tv_tensors import Mask

from anomalib.data.dataclasses.torch import ImageItem
from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import (
    DownloadInfo,
    LabelName,
    Split,
    read_image,
    read_mask,
    validate_path,
)
from anomalib.data.utils.dataframe import AnomalibDataFrame

logger = logging.getLogger(__name__)


IMG_EXTENSIONS = (".png", ".PNG")

DOWNLOAD_INFO = DownloadInfo(
    name="mvtec_loco",
    url="https://www.mydrive.ch/shares/48237/1b9106ccdfbb09a0c414bd49fe44a14a/download/430647091-1646842701"
    "/mvtec_loco_anomaly_detection.tar.xz",
    hashsum="9e7c84dba550fd2e59d8e9e231c929c45ba737b6b6a6d3814100f54d63aae687",
)

CATEGORIES = (
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
)


class MVTecLOCODataset(AnomalibDataset):
    """MVTec LOCO dataset class.

    Dataset class for loading and processing MVTec LOCO AD dataset images. Supports
    classification, detection and segmentation tasks.

    Args:
        root (Path | str): Path to root directory containing the dataset.
            Defaults to ``"./datasets/MVTec_LOCO"``.
        category (str): Category name, must be one of ``CATEGORIES``.
            Defaults to ``"breakfast_box"``.
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from anomalib.data.datasets import MVTecLocoDataset
        >>> dataset = MVTecLocoDataset(
        ...     root="./datasets/MVTec_LOCO",
        ...     category="breakfast_box",
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

        For detection tasks, samples include boxes:

        >>> dataset.task = "detection"
        >>> sample = dataset[0]
        >>> list(sample.keys())
        ['image_path', 'label', 'image', 'mask_path', 'mask', 'boxes']

        Images are PyTorch tensors with shape ``(C, H, W)``, masks have shape
        ``(H, W)``:

        >>> sample["image"].shape, sample["mask"].shape
        (torch.Size([3, 256, 256]), torch.Size([256, 256]))
    """

    def __init__(
        self,
        root: Path | str = "./datasets/MVTec_LOCO",
        category: str = "breakfast_box",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.root_category = Path(root) / category
        self.category = category
        self.split = split
        self.samples = make_dataset(
            self.root_category,
            split=self.split,
            extensions=IMG_EXTENSIONS,
        )

    def __getitem__(self, index: int) -> ImageItem:
        """Get a dataset item.

        Args:
            index (int): Index of the item to get.

        Returns:
            ImageItem: The dataset item.
        """
        row = self.samples.row_as_dict(index)
        image_path = row["image_path"]
        mask_path = row["mask_path"]
        label_index = row["label_index"]

        image = read_image(image_path, as_tensor=True)
        item = {"image_path": image_path, "gt_label": label_index}

        # Some of the categories in MVTec LOCO have multiple masks for the same image.
        if isinstance(mask_path, str):
            mask_path = [mask_path]

        # Only Anomalous (1) images have masks in anomaly datasets
        # Therefore, create empty mask for Normal (0) images.
        semantic_mask = (
            Mask(torch.zeros(image.shape[-2:], dtype=torch.bool))
            if label_index == LabelName.NORMAL
            else Mask(torch.stack([read_mask(path, as_tensor=True) for path in mask_path]))
        )

        binary_mask = Mask(semantic_mask.view(-1, *semantic_mask.shape[-2:]).any(dim=0))
        item["image"], item["gt_mask"] = (
            self.augmentations(image, binary_mask) if self.augmentations else (image, binary_mask)
        )

        item["mask_path"] = mask_path
        # List of masks with the original size for saturation based metrics calculation
        item["semantic_mask"] = semantic_mask

        return ImageItem(
            image=item["image"],
            gt_mask=item["gt_mask"],
            gt_label=torch.tensor(label_index),
            image_path=image_path,
            mask_path=item["mask_path"][0] if isinstance(item["mask_path"], list) else item["mask_path"],
        )


def make_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] = IMG_EXTENSIONS,
) -> AnomalibDataFrame:
    """Create MVTec LOCO AD samples by parsing the original MVTec LOCO AD data file structure.

    The files are expected to follow the structure:
        path/to/dataset/split/category/image_filename.png
        path/to/dataset/ground_truth/category/image_filename/000.png

    where there can be multiple ground-truth masks for the corresponding anomalous images.

    This function creates a dataframe to store the parsed information based on the following format:

    +---+---------------+-------+---------+-------------------------+-----------------------------+-------------+
    |   | path          | split | label   | image_path              | mask_path                  | label_index |
    +===+===============+=======+=========+===============+=======================================+=============+
    | 0 | datasets/name | test  | defect  | path/to/image/file.png  | [path/to/masks/file.png]    | 1           |
    +---+---------------+-------+---------+-------------------------+-----------------------------+-------------+

    Args:
        root (str | Path): Path to dataset
        split (str | Split | None): Dataset split (ie., either train or test).
            Defaults to ``None``.
        extensions (Sequence[str]): List of file extensions to be included in the dataset.
            Defaults to ``None``.

    Returns:
        DataFrame: an output dataframe containing the samples of the dataset.

    Examples:
        The following example shows how to get test samples from MVTec LOCO AD pushpins category:

        >>> root = Path('./MVTec_LOCO')
        >>> category = 'pushpins'
        >>> path = root / category
        >>> samples = make_mvtec_loco_dataset(path, split='test')
        >>> samples.head()
           path                split label image_path           mask_path label_index
        0  datasets/MVTec_LOCO/pushpins test good  [...]/good/105.png           0
        1  datasets/MVTec_LOCO/pushpins test good  [...]/good/017.png           0

    Raises:
        RuntimeError: If no valid images are found
        MisMatchError: If anomalous images and masks don't match
    """
    root = validate_path(root)

    # Retrieve the image and mask files
    samples_list = []
    for f in root.glob("**/*"):
        if f.suffix in extensions:
            parts = f.parts
            # 'ground_truth' and non 'ground_truth' path have a different structure
            if "ground_truth" not in parts:
                split_folder, label_folder, image_file = parts[-3:]
                image_path = f"{root}/{split_folder}/{label_folder}/{image_file}"
                samples_list.append((str(root), split_folder, label_folder, "", image_path))
            else:
                split_folder, label_folder, image_folder, image_file = parts[-4:]
                image_path = f"{root}/{split_folder}/{label_folder}/{image_folder}/{image_file}"
                samples_list.append((str(root), split_folder, label_folder, image_folder, image_path))

    if not samples_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = pl.DataFrame(samples_list, schema=["path", "split", "label", "image_folder", "image_path"], orient="row")

    # Replace validation to Split.VAL.value in the split column
    samples = samples.with_columns(
        pl.col("split").str.replace("validation", Split.VAL.value).alias("split"),
    )

    # Create label index for normal (0) and anomalous (1) images.
    samples = samples.with_columns(
        pl.when(pl.col("label") == "good")
        .then(pl.lit(int(LabelName.NORMAL)))
        .otherwise(pl.lit(int(LabelName.ABNORMAL)))
        .alias("label_index"),
    )

    # separate ground-truth masks from samples
    mask_samples = samples.filter(pl.col("split") == "ground_truth").sort("image_path")
    samples = samples.filter(pl.col("split") != "ground_truth").sort("image_path")

    # Group masks and aggregate the path into a list
    mask_samples = mask_samples.group_by(["path", "split", "label", "image_folder"]).agg(
        pl.col("image_path").alias("mask_path"),
    )

    # assign mask paths to anomalous test images
    samples = samples.with_columns(pl.lit(None).cast(pl.List(pl.Utf8)).alias("mask_path"))
    abnormal_test = samples.with_row_index("_idx").filter(
        (pl.col("split") == "test") & (pl.col("label_index") == int(LabelName.ABNORMAL)),
    )
    if len(abnormal_test) > 0 and len(mask_samples) > 0:
        # Sort mask_samples to match the ordering of abnormal test images
        # Sort by (label, image_folder) to align with abnormal_test sorted by image_path
        mask_samples_sorted = mask_samples.sort(["label", "image_folder"])
        update = pl.DataFrame(
            {"_idx": abnormal_test["_idx"], "_mask_path": mask_samples_sorted["mask_path"][: len(abnormal_test)]},
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

    # validate that the right mask files are associated with the right test images
    abnormal_samples = samples.filter(pl.col("label_index") == int(LabelName.ABNORMAL))
    if len(abnormal_samples) > 0:
        for row in abnormal_samples.iter_rows(named=True):
            image_stem = Path(row["image_path"]).stem
            mask_paths = row["mask_path"]
            if isinstance(mask_paths, list) and mask_paths:
                mask_parent_stems = {Path(mp).parent.stem for mp in mask_paths}
                if image_stem not in mask_parent_stems:
                    msg = (
                        "Mismatch between anomalous images and ground truth masks. "
                        "Make sure the parent folder of the mask files in 'ground_truth' folder "
                        "follows the same naming convention as the anomalous images in the dataset "
                        "(e.g., image: '005.png', mask: '005/000.png')."
                    )
                    raise MisMatchError(msg)

    # infer the task type
    task = "classification" if not samples["mask_path"].is_not_null().any() else "segmentation"

    if split:
        samples = samples.filter(pl.col("split") == split)

    return AnomalibDataFrame(samples, attrs={"task": task})
