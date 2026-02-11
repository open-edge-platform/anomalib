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

from pathlib import Path

import pandas as pd
from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.utils import LabelName, Split, validate_path

# Image extensions used in Kaputt dataset
IMG_EXTENSIONS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")

# Material categories in Kaputt dataset (based on item_material field)
MATERIAL_CATEGORIES = (
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "styrofoam",
    "wood",
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
) -> DataFrame:
    """Create Kaputt samples by parsing the Parquet metadata files.

    The Kaputt dataset uses Parquet files containing metadata about each image:
        - capture_id: Unique identifier for the capture
        - defect: Boolean indicating if the image has a defect
        - major_defect: Boolean indicating if the defect is major
        - defect_types: List of defect type strings
        - item_material: Material category of the item

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

    # Define split mappings
    split_map = {
        Split.TRAIN: "train",
        Split.VAL: "validation",
        Split.TEST: "test",
        "train": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
    }

    # Determine which splits to load
    splits_to_load = [split_map.get(split, str(split))] if split is not None else ["train", "validation", "test"]

    all_samples = []

    for split_name in splits_to_load:
        # Load query parquet file
        query_parquet = root / "datasets" / f"query-{split_name}.parquet"
        if not query_parquet.exists():
            msg = f"Query parquet file not found: {query_parquet}"
            raise FileNotFoundError(msg)

        query_df = pd.read_parquet(query_parquet)

        # Process query images
        for _, row in query_df.iterrows():
            capture_id = row["capture_id"]

            # Determine if defective
            is_defective = row.get("defect", False)
            label = "abnormal" if is_defective else "normal"
            label_index = LabelName.ABNORMAL if is_defective else LabelName.NORMAL

            # Build image path
            image_path = (
                root / f"query-{image_type}" / "data" / split_name / "query-data" / image_type / f"{capture_id}.jpg"
            )

            # Build mask path for defective images
            mask_path = ""
            if is_defective:
                mask_path = str(
                    root / "query-mask" / "data" / split_name / "query-data" / "mask" / f"{capture_id}.png",
                )

            # Convert split name back to anomalib format
            anomalib_split = "val" if split_name == "validation" else split_name

            sample = {
                "split": anomalib_split,
                "label": label,
                "image_path": str(image_path),
                "mask_path": mask_path,
                "label_index": int(label_index),
                "capture_id": capture_id,
                "defect_types": row.get("defect_types", []),
                "item_material": row.get("item_material", ""),
            }
            all_samples.append(sample)

        # Optionally load reference images (always normal)
        if use_reference:
            ref_parquet = root / "datasets" / f"reference-{split_name}.parquet"
            if ref_parquet.exists():
                ref_df = pd.read_parquet(ref_parquet)

                for _, row in ref_df.iterrows():
                    capture_id = row["capture_id"]

                    # Reference images are always normal
                    image_path = (
                        root
                        / f"reference-{image_type}"
                        / "data"
                        / split_name
                        / "reference-data"
                        / image_type
                        / f"{capture_id}.jpg"
                    )

                    anomalib_split = "val" if split_name == "validation" else split_name

                    sample = {
                        "split": anomalib_split,
                        "label": "normal",
                        "image_path": str(image_path),
                        "mask_path": "",
                        "label_index": int(LabelName.NORMAL),
                        "capture_id": capture_id,
                        "defect_types": [],
                        "item_material": row.get("item_material", ""),
                    }
                    all_samples.append(sample)

    if not all_samples:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg)

    samples = pd.DataFrame(all_samples)

    # Sort by image path for consistency
    samples = samples.sort_values(by="image_path", ignore_index=True).reset_index(drop=True)

    # Infer the task type - if any masks exist, it's segmentation
    has_masks = (samples["mask_path"] != "").any()
    samples.attrs["task"] = "segmentation" if has_masks else "classification"

    return samples
