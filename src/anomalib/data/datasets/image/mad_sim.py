# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MAD-Sim Dataset.

This module provides a PyTorch Dataset implementation for the MAD-Sim dataset.
The dataset contains 20 LEGO object categories with normal training images,
normal and anomalous test images, and pixel-level segmentation masks for
Burrs, Missing, and Stains defects. Download and extract the dataset from
https://github.com/EricLee0224/PAD#23-mad-simulated-dataset before loading it.

License:
    The dataset is distributed under CC BY-NC-SA 4.0, as documented in the
    dataset directory description at https://github.com/EricLee0224/PAD.

Reference:
    Zhou et al., "PAD: A Dataset and Benchmark for Pose-agnostic Anomaly
    Detection", NeurIPS 2023. https://arxiv.org/abs/2310.07716.
"""

from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.utils import LabelName, Split, validate_path
from anomalib.data.utils.path import resolve_path_under_root
from anomalib.utils.path import resolve_dataset_root

CATEGORIES = (
    "01Gorilla",
    "02Unicorn",
    "03Mallard",
    "04Turtle",
    "05Whale",
    "06Bird",
    "07Owl",
    "08Sabertooth",
    "09Swan",
    "10Sheep",
    "11Pig",
    "12Zalika",
    "13Pheonix",
    "14Elephant",
    "15Parrot",
    "16Cat",
    "17Scorpion",
    "18Obesobeso",
    "19Bear",
    "20Puppy",
)


class MADSimDataset(AnomalibDataset):
    """MAD-Sim dataset class.

    Load a category from the extracted archive for anomaly segmentation.

    Args:
        root (Path | str | None): Dataset root containing the 20 category folders.
            Defaults to ``./datasets/MAD-Sim``.
        category (str): Official category name. Defaults to ``01Gorilla``.
        augmentations (Transform | None): Image and mask augmentations.
            Defaults to ``None``.
        split (str | Split | None): ``train``, ``test``, or ``None`` for both.
            Defaults to ``None``.

    Raises:
        ValueError: If the category or split is invalid.

    Example:
        >>> from anomalib.data.datasets import MADSimDataset
        >>> dataset = MADSimDataset(
        ...     root="./datasets/MAD-Sim",
        ...     category="01Gorilla",
        ...     split="test",
        ... )
    """

    def __init__(
        self,
        root: Path | str | None = "./datasets/MAD-Sim",
        category: str = "01Gorilla",
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        super().__init__(augmentations=augmentations)
        if category not in CATEGORIES:
            msg = f"Unknown MAD-Sim category {category!r}. Available categories: {CATEGORIES}"
            raise ValueError(msg)
        self.category = category
        self.split = split
        self.root_category = resolve_path_under_root(resolve_dataset_root(root, "MAD-Sim"), category)
        self.samples = make_mad_sim_dataset(root=self.root_category, split=self.split)


def make_mad_sim_dataset(root: str | Path, split: str | Split | None = None) -> DataFrame:
    """Read official split folders and match masks by defect type and image stem.

    Args:
        root (str | Path): Category directory containing ``train``, ``test`` and
            ``ground_truth`` folders.
        split (str | Split | None): Split to load, or ``None`` for both splits.

    Returns:
        DataFrame: Image paths, labels, split names and mask paths, with the
        ``task`` attribute set to ``segmentation``.

    Raises:
        ValueError: If the split is invalid or a path escapes the category root.
        FileNotFoundError: If a required directory or anomaly mask is missing.
        RuntimeError: If a requested split contains no PNG images.
    """
    if split not in {None, Split.TRAIN, Split.TEST}:
        msg = f"Unknown MAD-Sim split {split!r}. Expected 'train', 'test', or None."
        raise ValueError(msg)
    root = validate_path(root)
    samples_list = []
    for subset in (Split.TRAIN, Split.TEST) if split is None else (Split(split),):
        split_root = resolve_path_under_root(root, subset.value)
        images = sorted(path for path in split_root.glob("*/*.png") if path.is_file())
        if not images:
            msg = f"Found 0 images in {split_root}"
            raise RuntimeError(msg)
        for image_path in images:
            label = image_path.parent.name
            allowed_labels = ("good",) if subset == Split.TRAIN else ("good", "Burrs", "Missing", "Stains")
            if label not in allowed_labels:
                msg = f"Unknown MAD-Sim label {label!r} in {subset.value}. Expected {allowed_labels}."
                raise ValueError(msg)
            label_index = LabelName.NORMAL if label == "good" else LabelName.ABNORMAL
            mask_path = ""
            if label_index == LabelName.ABNORMAL:
                mask_path = str(
                    resolve_path_under_root(root, Path("ground_truth") / label / f"{image_path.stem}_mask.png"),
                )
            samples_list.append({
                "image_path": str(validate_path(image_path, base_dir=root)),
                "split": subset.value,
                "label": label,
                "label_index": int(label_index),
                "mask_path": mask_path,
            })
    samples = DataFrame(samples_list)
    samples.attrs["task"] = "segmentation"
    return samples
