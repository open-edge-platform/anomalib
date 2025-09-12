# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Numpy-based dataclasses for Anomalib.

This module provides numpy-based implementations of the generic dataclasses used in
Anomalib. These classes are designed to work with :class:`numpy.ndarray` objects
for efficient data handling and processing in anomaly detection tasks.

The module contains two main classes:
    - :class:`NumpyItem`: For single data items
    - :class:`NumpyBatch`: For batched data items
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from anomalib.data.dataclasses.generic import _GenericBatch, _GenericItem


@dataclass
class NumpyItem(_GenericItem[np.ndarray, np.ndarray, np.ndarray, str]):
    """Dataclass for a single item in Anomalib datasets using numpy arrays.

    This class extends :class:`_GenericItem` for numpy-based data representation.
    It includes both input data (e.g., images, labels) and output data (e.g.,
    predictions, anomaly maps) as numpy arrays.

    The class uses the following type parameters:
        - Image: :class:`numpy.ndarray`
        - Label: :class:`numpy.ndarray`
        - Mask: :class:`numpy.ndarray`
        - Path: :class:`str`

    This implementation is suitable for numpy-based processing pipelines in
    Anomalib where GPU acceleration is not required.
    """


@dataclass
class NumpyBatch(_GenericBatch[np.ndarray, np.ndarray, np.ndarray, list[str]]):
    """Dataclass for a batch of items in Anomalib datasets using numpy arrays.

    This class extends :class:`_GenericBatch` for batches of numpy-based data.
    It represents multiple data points for batch processing in anomaly detection
    tasks.

    The class uses the following type parameters:
        - Image: :class:`numpy.ndarray` with shape ``(B, C, H, W)``
        - Label: :class:`numpy.ndarray` with shape ``(B,)``
        - Mask: :class:`numpy.ndarray` with shape ``(B, H, W)``
        - Path: :class:`list` of :class:`str`

    Where ``B`` represents the batch dimension that is prepended to all
    tensor-like fields.
    """

    def keys(self, include_none: bool = True) -> list[str]:
        """Return a list of field names in the NumpyBatch.

        Args:
            include_none: If True, returns all possible field names including those
                that are None. If False, returns only field names that have non-None values.
                Defaults to True for backward compatibility.

        Returns:
            List of field names that can be accessed on this NumpyBatch instance.
            When include_none=True, includes all fields from the input, output, and any
            additional field classes that the specific batch type inherits from.
            When include_none=False, includes only fields with actual data.

        Example:
            >>> batch = NumpyBatch(image=np.random.rand(2, 224, 224, 3))
            >>> all_keys = batch.keys()  # Default: include_none=True
            >>> 'pred_score' in all_keys  # True (even though it's None)
            True
            >>> set_keys = batch.keys(include_none=False)
            >>> 'pred_score' in set_keys  # False (because it's None)
            False
        """
        from dataclasses import fields

        if include_none:
            return [field.name for field in fields(self)]

        return [field.name for field in fields(self) if getattr(self, field.name) is not None]

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Get a field value using dictionary-like syntax.

        Args:
            key: Field name to access.

        Returns:
            The value of the specified field.

        Raises:
            KeyError: If the field name is not found.

        Example:
            >>> batch = NumpyBatch(image=np.random.rand(2, 224, 224, 3))
            >>> batch["image"].shape
            (2, 224, 224, 3)
            >>> batch["gt_label"]
            None
        """
        if not hasattr(self, key):
            msg = f"Field '{key}' not found in {self.__class__.__name__}"
            raise KeyError(msg)
        return getattr(self, key)
