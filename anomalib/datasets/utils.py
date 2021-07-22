"""
Dataset Utils
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision
from torch import Size, Tensor, nn


class Tiler:
    """
    Tile Image.
    """

    def __init__(self, tile_size: int, padding: int = 0, dilation: int = 1):
        self.tile_size = tile_size
        self.dilation = dilation
        self.padding = padding
        self.stride = tile_size

        self.batch_size: int
        self.image_size: Size

    def tile_image(self, image: Tensor) -> Tensor:
        """
        Split an image into tiles.

        Args:
            image: Input image

        Returns:
            Tiles from the original input image.

        """

        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        self.image_size = image.shape[2:]
        num_channels = image.shape[1]

        # Image tiles of size NxF**2xP, where F: tile size, P: # of tiles.
        image_tiles = nn.Unfold(self.tile_size, self.dilation, self.padding, self.stride)(image)

        # Permute dims to have the following dim: NxPxF**2
        image_tiles = image_tiles.permute(0, 2, 1)

        # converted tensor into NxPxHXW, Reshape tiles into PxCxFxF
        image_tiles = image_tiles.reshape(image_tiles.shape[1], num_channels, self.tile_size, self.tile_size)

        return image_tiles

    def tile_batch(self, batch: Tensor) -> Tensor:
        """
        Split Image Batch into tiles

        Args:
            batch (Tensor): Batch of images with NxCxHxW dims.

        Returns:
            Tensor: Tiles of batch of images.
        """

        self.batch_size = batch.shape[0]
        self.image_size = batch.shape[2:]

        batch_tiles_list: List[Tensor] = [self.tile_image(image) for image in batch]
        batch_tiles: Tensor = torch.cat(batch_tiles_list, dim=0)

        return batch_tiles

    def untile_image(
        self,
        tiles: Tensor,
        padding: int = 0,
        normalize: bool = True,
        pixel_range: Optional[tuple] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> Tensor:
        """
        Merge the tiles to form the original image.
        Args:
            tiles: Tiles to merge (stitch)
            padding: Number of pixels to skip when stitching the tiles.
            normalize: Normalize the output image.
            pixel_range: Pixel range of the output image.
            scale_each: Scale each tile before merging.
            pad_value: Pixel value of the pads between tiles.

        Returns:
            Output image by merging (stitching) the tiles.

        """

        _, img_width = self.image_size
        num_rows = img_width // self.tile_size

        grid = torchvision.utils.make_grid(tiles, num_rows, padding, normalize, pixel_range, scale_each, pad_value)

        return grid

    def untile_batch(
        self,
        tiles: Tensor,
        padding: int = 0,
        normalize: bool = True,
        pixel_range: Optional[tuple] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> Tensor:
        """
        Merge the tiles to form the original batch.
        Args:
            tiles: Tiles to merge (stitch)
            padding: Number of pixels to skip when stitching the tiles.
            normalize: Normalize the output image.
            pixel_range: Pixel range of the output image.
            scale_each: Scale each tile before merging.
            pad_value: Pixel value of the pads between tiles.

        Returns:
            Output image by merging (stitching) the tiles.

        """

        batch_list: List[Tensor] = []
        batch_tiles = torch.chunk(input=tiles, chunks=self.batch_size, dim=0)

        for image_tiles in batch_tiles:
            image = self.untile_image(image_tiles, padding, normalize, pixel_range, scale_each, pad_value)

            if len(image.shape) == 3:
                image = image.unsqueeze(0)

            batch_list.append(image)

        batch = torch.cat(batch_list, dim=0)

        return batch

    def __call__(self, batch: Tensor) -> Tensor:
        return self.tile_batch(batch)


class Denormalize:
    """
    Denormalize Torch Tensor into np image format.
    """

    def __init__(self, mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
        # If no mean and std provided, assign ImageNet values.
        if mean is None:
            mean = [0.485, 0.456, 0.406]

        if std is None:
            std = [0.229, 0.224, 0.225]

        self.mean = Tensor(mean)
        self.std = Tensor(std)

    def __call__(self, tensor: Tensor) -> np.ndarray:
        """
        Denormalize the input

        Args:
            tensor: Input tensor image (C, H, W)

        Returns:
            Denormalized numpy array (H, W, C).

        """

        if tensor.dim() == 4:
            if tensor.size(0):
                tensor = tensor.squeeze(0)
            else:
                raise ValueError(f"Tensor has batch size of {tensor.size(0)}. Only single batch is supported.")

        for tnsr, mean, std in zip(tensor, self.mean, self.std):
            tnsr.mul_(std).add_(mean)

        array = (tensor * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return array

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ToNumpy:
    """
    Convert Tensor into Numpy Array
    """

    def __call__(self, tensor: Tensor, dims: Optional[Tuple[int, ...]] = None) -> np.ndarray:

        # Default support is (C, H, W) or (N, C, H, W)
        if dims is None:
            dims = (0, 2, 3, 1) if len(tensor.shape) == 4 else (1, 2, 0)

        array = (tensor * 255).permute(dims).cpu().numpy().astype(np.uint8)

        if array.shape[0] == 1:
            array = array.squeeze(0)
        if array.shape[-1] == 1:
            array = array.squeeze(-1)

        return array

    def __repr__(self):
        return self.__class__.__name__ + "()"
