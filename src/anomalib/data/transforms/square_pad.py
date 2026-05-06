# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Square padding transform for Anomalib.

This module provides a square padding transform that pads the shorter side of an
image to match the longer side using edge replication, preserving aspect ratio
before resizing.

Example:
    >>> import torch
    >>> from anomalib.data.transforms.square_pad import SquarePad
    >>> transform = SquarePad()
    >>> image = torch.randn(3, 200, 300)
    >>> output = transform(image)
    >>> output.shape
    torch.Size([3, 300, 300])
"""

from typing import Any

import torch
from torch.nn import functional as F  # noqa: N812
from torchvision.transforms.v2 import Transform


class SquarePad(Transform):
    """Pad a tensor to a square by replicating edge pixels.

    Pads the shorter spatial dimension so that height and width are equal.
    Images use replication padding (edge pixels are repeated). Boolean masks
    use constant padding with ``0`` since ``replication_pad2d`` does not
    support boolean tensors.

    Example:
        >>> import torch
        >>> transform = SquarePad()
        >>> image = torch.randn(3, 200, 300)
        >>> output = transform(image)
        >>> output.shape
        torch.Size([3, 300, 300])
    """

    def _transform(self, inpt: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:  # noqa: PLR6301
        """Apply square padding to the input tensor.

        Args:
            inpt (torch.Tensor): Input tensor to pad.
            params (dict[str, Any]): Transform parameters (unused).

        Returns:
            torch.Tensor: Square-padded tensor.
        """
        del params
        h, w = inpt.shape[-2:]
        max_side = max(h, w)
        pad_h = max_side - h
        pad_w = max_side - w
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        # Boolean masks (e.g. gt_mask) do not support replication padding.
        if inpt.dtype == torch.bool:
            return F.pad(inpt, padding, mode="constant", value=0)
        return F.pad(inpt, padding, mode="replicate")

    def transform(self, inpt: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
        """Wrapper for ``_transform``.

        Ensures compatibility with newer Torchvision versions where
        ``_transform`` was renamed to ``transform``.

        Args:
            inpt (torch.Tensor): Input tensor to pad.
            params (dict[str, Any]): Transform parameters (unused).

        Returns:
            torch.Tensor: Square-padded tensor.
        """
        return self._transform(inpt, params)
