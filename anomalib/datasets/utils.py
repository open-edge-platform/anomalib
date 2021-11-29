"""Dataset Utils."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import List, Optional, Tuple

import cv2
import numpy as np
from torch import Tensor


def read_image(path: str) -> np.ndarray:
    """Read image from disk in RGB format.

    Args:
        path (str): path to the image file

    Returns:
        image as numpy array
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


class Denormalize:
    """Denormalize Torch Tensor into np image format.

    Args:
        mean (Optional[List[float]], optional): Mean used for denormalizing. Defaults to None.
        std (Optional[List[float]], optional): Standard deviation used for denormalizing. Defaults to None.
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
        """Denormalize the input.

        Args:
            tensor (Tensor): Input tensor image (C, H, W)

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

    def __repr__(self) -> str:
        """Prints `Denormalize()`.

        Returns:
            (str): Return string with class name
        """
        return self.__class__.__name__ + "()"


class ToNumpy:
    """Convert Tensor into Numpy Array."""

    def __call__(self, tensor: Tensor, dims: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """Convert torch tensor to numpy.

        Args:
            tensor (Tensor): Input tensor in range 0-1
            dims (Optional[Tuple[int, ...]], optional): Tuple corresponding to axis permutation from torch tensor to
             numpy array. Defaults to None.

        Returns:
            np.ndarray: Converted numpy array in range 0-255
        """
        # Default support is (C, H, W) or (N, C, H, W)
        if dims is None:
            dims = (0, 2, 3, 1) if len(tensor.shape) == 4 else (1, 2, 0)

        array = (tensor * 255).permute(dims).cpu().numpy().astype(np.uint8)

        if array.shape[0] == 1:
            array = array.squeeze(0)
        if array.shape[-1] == 1:
            array = array.squeeze(-1)

        return array

    def __repr__(self) -> str:
        """Return `ToNumpy()`.

        Returns:
            str: Class name
        """
        return self.__class__.__name__ + "()"
