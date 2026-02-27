# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""BMAD Dataset.

This module provides a PyTorch Dataset implementation for the BMAD
(Benchmarks for Medical Anomaly Detection) dataset. The dataset will be
downloaded and prepared automatically if not found locally.

BMAD is a standardized benchmark for anomaly detection in medical imaging.
It reorganizes six public datasets across five medical domains:
brain MRI, liver CT, retinal OCT, retinal fundus (RESC), chest X-ray,
and digital histopathology. Among these, three datasets provide
pixel-level ground truth masks for anomaly localization, while the others
are for sample-level anomaly detection only.

License:
    BMAD is released under the Creative Commons
    Attribution-NonCommercial-ShareAlike 4.0 International License
    (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/

Reference:
    Bao, J., Sun, H., Deng, H., He, Y., Zhang, Z., & Li, X. (2023).
    BMAD: Benchmarks for Medical Anomaly Detection.
    arXiv preprint arXiv:2306.11876.
    https://doi.org/10.48550/arXiv.2306.11876
"""

from pathlib import Path

import polars as pl
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path
from anomalib.data.utils.dataframe import AnomalibDataFrame

CATEGORIES = ("Brain", "Chest", "Histopathology", "Liver", "Retina_OCT2017", "Retina_RESC")


class BMADDataset(AnomalibDataset):
    """BMAD dataset class.

    Dataset class for loading and processing the BMAD (Benchmarks for Medical
    Anomaly Detection) dataset. Supports both classification and segmentation
    tasks depending on the modality. Some categories (e.g. brain MRI, liver CT,
    and histopathology) include pixel-level ground truth masks for segmentation,
    while others (e.g. chest X-ray, retinal OCT, and retinal fundus) are
    classification only.

    Args:
        root (Path | str): Path to root directory containing the dataset.
            Defaults to ``"./datasets/BMAD"``.
        category (str): Category name, must be one of
            ``["Brain", "Liver", "Retina_OCT2017", "Retina_RESC", "Chest", "Histopathology"]``.
            Defaults to ``"Brain"``.
        augmentations (Transform, optional): Augmentations that should be applied
            to the input images. Defaults to ``None``.
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import BMADDataset
        >>> dataset = BMADDataset(
        ...     root=Path("./datasets/BMAD"),
        ...     category="Brain",
        ...     split="train"
        ... )

        For classification-only tasks, each sample contains:

        >>> sample = dataset[0]
        >>> list(sample.keys())
        ['image_path', 'label', 'image']

        For segmentation tasks (if the dataset includes pixel-level annotations),
        samples also include mask paths and masks:

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
        root: str | Path,
        category: str,
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.root_category = Path(root) / category
        self.split = split
        self.samples = make_bmad_dataset(path=self.root_category, split=self.split)


def make_bmad_dataset(path: Path, split: str | Split | None = None) -> AnomalibDataFrame:
    """Create BMAD samples by parsing the dataset structure.

    The files are expected to follow the structure:

    .. code-block:: text

        path/to/dataset
        ├── category
        │   ├── train
        │   │   ├── good
        │   │   │   ├── image_filename.png
        │   │   ├── Ungood
        │   │   │   ├── img
        │   │   │   │   ├── image_filename.png
        │   │   │   │   └── label
        │   │   │   │       └── image_filename.png
        │   │   └── test
        │   │       ├── good
        │   │       │   ├── img
        │   │       │   │   ├── image_filename.png
        │   │       │   │   └── label
        │   │       │   │       └── image_filename.png
        │   │       └── Ungood
        │   │           ├── img
        │   │           │   ├── image_filename.png
        │   │           │   └── label


    Args:
        path (Path | str): Path to dataset root directory.
        split (str | Split | None, optional): Dataset split (train/test/val). Defaults to ``None``.

    Returns:
        DataFrame with columns:
            - path: Dataset base path
            - domain: Medical domain (e.g. brain, liver, retina, chest, histopathology)
            - split: Dataset split
            - label: Class label
            - image_path: Path to image
            - mask_path: Path to mask (if any)
            - label_index: Numeric label (0=normal, 1=abnormal)

    Example:
        >>> samples = make_bmad_dataset(Path("./datasets/BMAD/brain_mri"), split="train")
        >>> samples.head()

    Raises:
        RuntimeError: If no valid images are found.
        MisMatchError: If anomalous images and masks don't align.
    """
    path = validate_path(path)

    samples_list = [
        (
            (str(path), *filename.parts[-4:])
            if filename.parts[-3] != "train"
            else (str(path), *filename.parts[-3:-1], "", filename.parts[-1])
        )
        for filename in path.glob("**/*")
        if filename.suffix in {".png", ".PNG"}
    ]

    samples = pl.DataFrame(samples_list, schema=["path", "split", "label", "temp", "image_path"], orient="row")

    samples = samples.with_columns(
        (
            pl.col("path")
            + "/"
            + pl.col("split")
            + "/"
            + pl.col("label")
            + "/"
            + pl.when(pl.col("temp") != "").then(pl.col("temp") + "/").otherwise(pl.lit(""))
            + pl.col("image_path")
        ).alias("image_path"),
    )

    samples = samples.with_columns(
        pl.when(pl.col("label") == "good")
        .then(pl.lit(int(LabelName.NORMAL)))
        .otherwise(pl.lit(int(LabelName.ABNORMAL)))
        .alias("label_index"),
    )

    mask_samples = samples.filter(pl.col("temp") == "label").sort("image_path")
    samples = samples.filter(pl.col("temp") != "label").sort("image_path")

    samples = samples.with_columns(pl.lit("").alias("mask_path"))
    if len(mask_samples) > 0:
        abnormal_masks = mask_samples.filter(pl.col("label_index") == int(LabelName.ABNORMAL))
        abnormal_test_valid = samples.with_row_index("_idx").filter(
            ((pl.col("split") == "test") | (pl.col("split") == "valid"))
            & (pl.col("label_index") == int(LabelName.ABNORMAL)),
        )
        if len(abnormal_test_valid) > 0 and len(abnormal_masks) > 0:
            update = pl.DataFrame(
                {
                    "_idx": abnormal_test_valid["_idx"],
                    "_mask_path": abnormal_masks["image_path"][: len(abnormal_test_valid)],
                },
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

    if len(mask_samples) > 0:
        abnormal_samples = samples.filter(pl.col("label_index") == int(LabelName.ABNORMAL))
        if len(abnormal_samples) > 0:
            for row in abnormal_samples.iter_rows(named=True):
                if row["mask_path"] and Path(row["image_path"]).stem not in Path(row["mask_path"]).stem:
                    msg = (
                        "Mismatch between anomalous images and ground truth masks. Make sure "
                        "mask files in 'Ungood/label/' folder follow the same naming "
                        "convention as the anomalous images (e.g. image: '000.png', "
                        "mask: '000.png')."
                    )
                    raise MisMatchError(msg)

    task = "classification" if (samples["mask_path"] == "").all() else "segmentation"
    if split:
        samples = samples.filter(pl.col("split") == split)

    return AnomalibDataFrame(samples, attrs={"task": task})
