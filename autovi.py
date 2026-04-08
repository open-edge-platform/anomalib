# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AutoVI Dataset.

AutoVI is a genuine industrial production dataset for unsupervised anomaly detection.
This dataset can be found at https://zenodo.org/records/10459003.
This loader only supports AutoVI's first edition (1.0.0).

License:
Copyright © 2023-2024 Renault Group

This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
To view a copy of the license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/.
For using the data in a way that falls under the commercial use clause of the license, please contact us.

Attribution:
Please use the following for citing the dataset in scientific work:
Carvalho, P., Lafou, M., Durupt, A., Leblanc, A., & Grandvalet, Y. (2024).
The Automotive Visual Inspection Dataset (AutoVI):
A Genuine Industrial Production Dataset for Unsupervised Anomaly Detection [Dataset].
https://doi.org/10.5281/zenodo.10459003

This module provides PyTorch Dataset implementation for datasets with the custom
structure where masks are nested in numbered folders.

Dataset Structure:
    root/
    ├── train/
    │   └── good/
    │       └── *.png
    ├── test/
    │   ├── good/
    │   │   └── *.png
    │   └── defect_type/
    │       └── *.png
    └── ground_truth/
        └── defect_type/
            └── ####/
                └── ####.png


Adapted from MVTec AD dataset loader
"""

from collections.abc import Sequence
from pathlib import Path

from pandas import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path

IMG_EXTENSIONS = (".png", ".PNG")


class AutoVIDataset(AnomalibDataset):
    """AutoVI Anomaly Detection dataset class.

    Dataset class for loading datasets with the custom structure where masks
    are nested in numbered folders. Supports both classification and segmentation tasks.

    Args:
        root (Path | str): Path to root directory containing the dataset.
            Example: "C:/Users/joe/datasets/engine_wiring"
        augmentations (Transform, optional): Augmentations that should be applied to the input images.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split - usually
            ``Split.TRAIN`` or ``Split.TEST``. Defaults to ``None``.

    Example:
        >>> from pathlib import Path
        >>> from custom_dataset import AutoVIDataset
        >>> dataset = AutoVIDataset(
        ...     root=Path("C:/Users/joe/datasets/engine_wiring"),
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
    """

    def __init__(
        self,
        root: Path | str,
        augmentations: Transform | None = None,
        split: str | Split | None = None,
    ) -> None:
        """Initialize the AutoVI anomaly dataset.

        Args:
            root: Root directory of the dataset (e.g., engine_wiring folder)
            augmentations: Optional image transformations
            split: Dataset split ('train' or 'test')
        """
        super().__init__(augmentations=augmentations)

        self.root = Path(root)
        self.split = split
        self.samples = make_custom_dataset(
            self.root,
            split=self.split,
            extensions=IMG_EXTENSIONS,
        )


def _collect_train_test_images(root: Path, extensions: Sequence[str]) -> list[tuple[str, str, str, str]]:
    """Collect train and test images from the dataset structure.

    Args:
        root: Path to dataset root directory
        extensions: Valid file extensions

    Returns:
        List of tuples: (root, split, label, filename)
    """
    train_test_samples = []
    for split_dir in ["train", "test"]:
        split_path = root / split_dir
        if split_path.exists():
            for img_file in split_path.glob("**/*"):
                if img_file.suffix in extensions and img_file.is_file():
                    parts = img_file.relative_to(root).parts
                    if len(parts) >= 3:  # split/label/filename
                        train_test_samples.append((
                            str(root),
                            parts[0],  # split (train/test)
                            parts[1],  # label (good/defect_type)
                            img_file.name,  # filename
                        ))
    return train_test_samples


def _collect_mask_samples(root: Path, extensions: Sequence[str]) -> list[tuple[str, str, str, str, str]]:
    """Collect mask samples from ground_truth directory.

    Args:
        root: Path to dataset root directory
        extensions: Valid file extensions

    Returns:
        List of tuples: (root, split, defect_type, mask_id, mask_path)
    """
    mask_samples_list = []
    ground_truth_path = root / "ground_truth"

    if ground_truth_path.exists():
        for mask_file in ground_truth_path.glob("**/*"):
            if mask_file.suffix in extensions and mask_file.is_file():
                parts = mask_file.relative_to(ground_truth_path).parts
                if len(parts) >= 3:  # defect_type/####/####.png
                    mask_samples_list.append((
                        str(root),
                        "ground_truth",
                        parts[0],  # defect_type
                        parts[1],  # mask folder number
                        str(mask_file),  # full path to mask
                    ))
    return mask_samples_list


def _create_samples_dataframe(train_test_samples: list[tuple[str, str, str, str]]) -> DataFrame:
    """Create DataFrame from train/test samples and add label indices.

    Args:
        train_test_samples: List of (root, split, label, filename) tuples

    Returns:
        DataFrame with image paths and label indices
    """
    samples = DataFrame(
        train_test_samples,
        columns=["path", "split", "label", "image_path"],
    )

    # Build full image paths
    samples["image_path"] = samples.path + "/" + samples.split + "/" + samples.label + "/" + samples.image_path

    # Create label index
    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    return samples


def _match_masks_to_images(samples: DataFrame, mask_samples: DataFrame) -> None:
    """Match masks to anomalous test images in-place.

    Args:
        samples: DataFrame with image samples
        mask_samples: DataFrame with mask samples
    """
    samples["mask_path"] = ""

    if mask_samples.empty:
        return

    for idx, row in samples.iterrows():
        if row["split"] == "test" and row["label_index"] == LabelName.ABNORMAL:
            img_stem = Path(row["image_path"]).stem
            matching_masks = mask_samples[
                (mask_samples["label"] == row["label"]) & (mask_samples["mask_id"] == img_stem)
            ]
            if not matching_masks.empty:
                samples.loc[idx, "mask_path"] = matching_masks.iloc[0]["mask_path"]


def _verify_mask_matching(samples: DataFrame) -> None:
    """Verify that masks are correctly matched to images.

    Args:
        samples: DataFrame with matched samples

    Raises:
        MisMatchError: If mask names don't match image names
    """
    abnormal_samples = samples.loc[(samples.label_index == LabelName.ABNORMAL) & (samples.split == "test")]

    if len(abnormal_samples) == 0:
        return

    # Check for missing masks
    missing_masks = abnormal_samples[abnormal_samples["mask_path"] == ""]
    if not missing_masks.empty:
        print(f"Warning: {len(missing_masks)} abnormal test images have no matching masks")
        print("Examples of images without masks:")
        for _idx, row in missing_masks.head().iterrows():
            print(f"  - {row['image_path']}")

    # Verify mask names match image names
    samples_with_masks = abnormal_samples[abnormal_samples["mask_path"] != ""]
    if len(samples_with_masks) > 0:
        valid_matches = samples_with_masks.apply(
            lambda x: Path(x.image_path).stem in str(Path(x.mask_path)),
            axis=1,
        )

        if not valid_matches.all():
            msg = (
                "Mismatch between anomalous images and ground truth masks. "
                "Mask files in 'ground_truth/defect_type/####/' folders should "
                "have the same base name as the corresponding test images "
                "(e.g., test image: 'test/blue_hoop/0000.png', "
                "mask: 'ground_truth/blue_hoop/0000/0000.png')."
            )
            raise MisMatchError(msg)


def make_custom_dataset(
    root: str | Path,
    split: str | Split | None = None,
    extensions: Sequence[str] | None = None,
) -> DataFrame:
    """Create custom dataset samples by parsing the data directory structure.

    The files are expected to follow the structure:
        ``root/train/good/image.png``
        ``root/test/good/image.png``
        ``root/test/defect_type/image.png``
        ``root/ground_truth/defect_type/####/####.png``

    Args:
        root (Path | str): Path to dataset root directory
        split (str | Split | None, optional): Dataset split (train or test)
            Defaults to ``None``.
        extensions (Sequence[str] | None, optional): Valid file extensions
            Defaults to ``None``.

    Returns:
        DataFrame: Dataset samples with columns:
            - path: Base path to dataset
            - split: Dataset split (train/test)
            - label: Class label (defect type or 'good')
            - image_path: Path to image file
            - mask_path: Path to mask file (if available)
            - label_index: Numeric label (0=normal, 1=abnormal)

    Example:
        >>> root = Path("C:/Users/joe/datasets/engine_wiring")
        >>> samples = make_custom_dataset(root, split="train")
        >>> samples.head()

    Raises:
        RuntimeError: If no valid images are found
        MisMatchError: If anomalous images and masks don't match
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS

    root = validate_path(root)
    root = Path(root)

    # Collect train and test images
    train_test_samples = _collect_train_test_images(root, extensions)
    if not train_test_samples:
        msg = f"Found 0 images in {root}/train and {root}/test"
        raise RuntimeError(msg)
    print(f"Found {len(train_test_samples)} images in {root}/train and {root}/test")

    # Create samples DataFrame
    samples = _create_samples_dataframe(train_test_samples)

    # Collect and process masks
    mask_samples_list = _collect_mask_samples(root, extensions)
    if not mask_samples_list:
        msg = f"Found 0 images in {root}/ground_truth"
        raise RuntimeError(msg)
    print(f"Found {len(mask_samples_list)} images in {root}/ground_truth")

    mask_samples = DataFrame(
        mask_samples_list,
        columns=["path", "split", "label", "mask_id", "mask_path"],
    )
    mask_samples = mask_samples.sort_values(by=["label", "mask_id"], ignore_index=True)

    # Match masks to images and verify
    _match_masks_to_images(samples, mask_samples)
    _verify_mask_matching(samples)

    # Infer task type
    has_masks = (samples["mask_path"] != "").any()
    samples.attrs["task"] = "segmentation" if has_masks else "classification"

    # Filter by split if specified
    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples


def load_all_datasets(datasets_root: str | Path) -> dict[str, dict[str, AutoVIDataset]]:
    """Load all datasets from the datasets root directory.

    Args:
        datasets_root: Path to the datasets folder containing multiple dataset folders
                      (e.g., "C:/Users/joe/datasets")

    Returns:
        Dictionary mapping dataset names to AutoVIDataset instances

    Example:
        >>> datasets = load_all_datasets("C:/Users/joe/datasets")
        >>> engine_wiring_train = datasets["engine_wiring"]["train"]
        >>> engine_wiring_test = datasets["engine_wiring"]["test"]
    """
    datasets_root = Path(datasets_root)
    all_datasets = {}

    # Find all dataset folders (those containing train/test/ground_truth)
    for dataset_folder in datasets_root.iterdir():
        if dataset_folder.is_dir():
            # Check if it has the expected structure
            has_train = (dataset_folder / "train").exists()
            has_test = (dataset_folder / "test").exists()

            if has_train or has_test:
                dataset_name = dataset_folder.name
                all_datasets[dataset_name] = {
                    "train": AutoVIDataset(dataset_folder, split="train"),
                    "test": AutoVIDataset(dataset_folder, split="test"),
                    "all": AutoVIDataset(dataset_folder, split=None),
                }
                print(f"Loaded dataset: {dataset_name}")
                print(f"  Train samples: {len(all_datasets[dataset_name]['train'].samples)}")
                print(f"  Test samples: {len(all_datasets[dataset_name]['test'].samples)}")

    return all_datasets


if __name__ == "__main__":
    # Example usage
    datasets_root = Path("C:/Users/utcpret/perso/datasets")

    # Or load all datasets
    print("loading all datasets")
    all_datasets = load_all_datasets(datasets_root)
