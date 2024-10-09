"""Inference Dataset."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path

from torch.utils.data.dataset import Dataset

from anomalib.data import ImageBatch, ImageItem
from anomalib.data.utils import get_image_filenames, read_image


class PredictDataset(Dataset):
    """Inference Dataset to perform prediction.

    Args:
        path (str | Path): Path to an image or image-folder.
        image_size (int | tuple[int, int] | None, optional): Target image size
            to resize the original image. Defaults to None.
    """

    def __init__(
        self,
        path: str | Path,
        image_size: int | tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()

        self.image_filenames = get_image_filenames(path)
        self.image_size = image_size

    def __len__(self) -> int:
        """Get the number of images in the given path."""
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> ImageItem:
        """Get the image based on the `index`."""
        image_filename = self.image_filenames[index]
        image = read_image(image_filename, as_tensor=True)

        return ImageItem(image=image, image_path=str(image_filename))

    @property
    def collate_fn(self) -> Callable:
        """Get the collate function."""
        return ImageBatch.collate
