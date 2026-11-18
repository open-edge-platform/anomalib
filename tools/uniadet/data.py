# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Model-specific auxiliary data protocol; never replace the normal target training split."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import tv_tensors
from torchvision.transforms.v2 import (
    ColorJitter,
    Compose,
    RandomApply,
    RandomChannelPermutation,
    RandomChoice,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
    Resize,
)

from anomalib.data import AnomalibDataModule, ImageBatch, ImageItem
from anomalib.data.datasets.image.mvtecad import CATEGORIES as MVTEC_CATEGORIES
from anomalib.data.datasets.image.mvtecad import MVTecADDataset
from anomalib.data.datasets.image.visa import CATEGORIES as VISA_CATEGORIES
from anomalib.data.datasets.image.visa import VisaDataset
from anomalib.data.utils import Split


class AuxiliaryDataset(Dataset):
    """All auxiliary categories' labeled test images, with paper-style class-aware augmentation.

    This data is used exclusively for auxiliary head pretraining. Evaluation
    must use a different dataset; none of its test images belong in this loader.

    Args:
        root (Path): MVTec root or VisA's prepared ``visa_pytorch`` directory.
        dataset (str): ``"mvtec"`` or ``"visa"``.
        image_size (int): Square resize resolution.
        augmentation (bool): Enable training color/geometric transforms and grid mosaic/cropping with probability 0.5.

    Raises:
        ValueError: If the dataset name is unsupported.
    """

    def __init__(self, root: Path, dataset: str, image_size: int, augmentation: bool = True) -> None:
        if dataset not in {"mvtec", "visa"}:
            msg = "Auxiliary dataset must be 'mvtec' or 'visa'."
            raise ValueError(msg)
        categories = MVTEC_CATEGORIES if dataset == "mvtec" else VISA_CATEGORIES
        dataset_class = MVTecADDataset if dataset == "mvtec" else VisaDataset
        self.resize = Resize((image_size, image_size), antialias=False)
        self.datasets = [
            dataset_class(root=root, category=category, split=Split.TEST, augmentations=self.resize)
            for category in categories
        ]
        self.indices = [(category, index) for category, data in enumerate(self.datasets) for index in range(len(data))]
        self.normal_indices = [data.samples.index[data.samples.label_index == 0].tolist() for data in self.datasets]
        self.augmentation = augmentation
        self.augmentations = Compose([
            RandomApply([ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.5),
            RandomApply([RandomChoice([RandomRotation((angle, angle)) for angle in (0, 90, 180, 270)])], p=0.5),
            RandomApply([RandomChannelPermutation()], p=0.25),
            RandomGrayscale(p=0.25),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
        ])

    def __len__(self) -> int:
        """Return the number of auxiliary test images."""
        return len(self.indices)

    def __getitem__(self, index: int) -> ImageItem:
        """Load and optionally augment a sample without changing its category or image label.

        Args:
            index (int): Auxiliary sample index.

        Returns:
            ImageItem: Image, binary mask, and label.
        """
        category, local_index = self.indices[index]
        item = self._load_item(category, local_index)
        if not self.augmentation:
            return item
        if torch.rand(()) < 0.5:
            grid = int(torch.randint(2, 4, ()))
            item = self._mosaic(item, category, grid) if torch.rand(()) < 0.5 else self._crop(item, grid)
        image, mask = self._supervision(item)
        image, mask = self.augmentations(tv_tensors.Image(image), tv_tensors.Mask(mask))
        return ImageItem(image=image, gt_mask=mask, gt_label=item.gt_label)

    def _load_item(self, category: int, index: int) -> ImageItem:
        item = self.datasets[category][index]
        if not isinstance(item, ImageItem):
            msg = "Auxiliary supervision requires image items."
            raise TypeError(msg)
        return item

    @staticmethod
    def _supervision(item: ImageItem) -> tuple[torch.Tensor, torch.Tensor]:
        if item.image is None or item.gt_mask is None:
            msg = "Auxiliary supervision requires both an image and a pixel mask."
            raise ValueError(msg)
        return item.image, item.gt_mask

    def _mosaic(self, item: ImageItem, category: int, grid: int) -> ImageItem:
        candidates = self.normal_indices[category] if item.gt_label == 0 else list(range(len(self.datasets[category])))
        if not candidates:
            return item
        tiles = [item]
        for _ in range(grid * grid - 1):
            index = candidates[int(torch.randint(len(candidates), ()))]
            tiles.append(self._load_item(category, index))
        order = torch.randperm(len(tiles)).tolist()
        source_image, _ = self._supervision(item)
        tile_size = source_image.shape[-1] // grid
        resize = Resize((tile_size, tile_size), antialias=False)
        images, masks = [], []
        for index in order:
            image, mask = resize(*self._supervision(tiles[index]))
            images.append(image)
            masks.append(mask)
        image_rows = [torch.cat(images[start : start + grid], dim=-1) for start in range(0, len(images), grid)]
        mask_rows = [torch.cat(masks[start : start + grid], dim=-1) for start in range(0, len(masks), grid)]
        image, mask = self.resize(
            tv_tensors.Image(torch.cat(image_rows, dim=-2)),
            tv_tensors.Mask(torch.cat(mask_rows, dim=-2)),
        )
        return ImageItem(image=image, gt_mask=mask, gt_label=item.gt_label)

    def _crop(self, item: ImageItem, grid: int) -> ImageItem:
        source_image, source_mask = self._supervision(item)
        height, width = source_image.shape[-2:]
        crops = []
        for row in range(grid):
            for column in range(grid):
                top, bottom = row * height // grid, (row + 1) * height // grid
                left, right = column * width // grid, (column + 1) * width // grid
                mask = source_mask[top:bottom, left:right]
                if item.gt_label == 0 or mask.any():
                    crops.append((top, bottom, left, right))
        if not crops:
            return item
        top, bottom, left, right = crops[int(torch.randint(len(crops), ()))]
        image, mask = self.resize(
            tv_tensors.Image(source_image[:, top:bottom, left:right]),
            tv_tensors.Mask(source_mask[top:bottom, left:right]),
        )
        return ImageItem(image=image, gt_mask=mask, gt_label=item.gt_label)


class CalibrationData(AnomalibDataModule):
    """Disjoint auxiliary-domain calibration and validation subsets for NNCF.

    Target test data is never used for calibration or accuracy-control fitting.

    Args:
        dataset (Dataset): Labeled, unnormalized auxiliary images without augmentation.
        subset_size (int): Images in each of the two disjoint subsets.

    Raises:
        ValueError: If the two subsets do not fit within the dataset.
    """

    def __init__(self, dataset: Dataset, subset_size: int = 300) -> None:
        super().__init__(train_batch_size=1, eval_batch_size=1, num_workers=0)
        if subset_size <= 0 or 2 * subset_size > len(dataset):
            msg = "Calibration requires two disjoint subsets that fit within the auxiliary dataset."
            raise ValueError(msg)
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()
        self.calibration = Subset(dataset, indices[:subset_size])
        self.validation = Subset(dataset, indices[subset_size : 2 * subset_size])

    def _setup(self, _stage: str | None = None) -> None:
        """Subsets are created once at construction."""

    def setup(self, stage: str | None = None) -> None:
        """Keep calibration inputs raw; preprocessing is already inside the exported graph."""
        del stage
        self._is_setup = True

    def val_dataloader(self) -> DataLoader:
        """Return the quantization calibration subset."""
        return DataLoader(self.calibration, batch_size=1, collate_fn=ImageBatch.collate)

    def test_dataloader(self) -> DataLoader:
        """Return the separate accuracy-control validation subset."""
        return DataLoader(self.validation, batch_size=1, collate_fn=ImageBatch.collate)
