# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch-based dataclasses for Anomalib.

This module provides PyTorch-based implementations of the generic dataclasses used
in Anomalib. These classes are designed to work with PyTorch tensors for efficient
data handling and processing in anomaly detection tasks.

These classes extend the generic dataclasses defined in the Anomalib framework,
providing concrete implementations that use PyTorch tensors for tensor-like data.
"""

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Generic, NamedTuple, TypeVar

import torch
from torchvision.tv_tensors import Mask

from anomalib.data.dataclasses.generic import ImageT, _GenericBatch, _GenericItem

NumpyT = TypeVar("NumpyT")


class InferenceBatch(NamedTuple):
    """Batch for use in torch and inference models.

    Args:
        pred_score (torch.Tensor | None): Predicted anomaly scores.
            Defaults to ``None``.
        pred_label (torch.Tensor | None): Predicted anomaly labels.
            Defaults to ``None``.
        anomaly_map (torch.Tensor | None): Generated anomaly maps.
            Defaults to ``None``.
        pred_mask (torch.Tensor | None): Predicted anomaly masks.
            Defaults to ``None``.
    """

    pred_score: torch.Tensor | None = None
    pred_label: torch.Tensor | None = None
    anomaly_map: torch.Tensor | None = None
    pred_mask: torch.Tensor | None = None


@dataclass
class ToNumpyMixin(Generic[NumpyT]):
    """Mixin for converting torch-based dataclasses to numpy.

    This mixin provides functionality to convert PyTorch tensor data to numpy
    arrays. It requires the subclass to define a ``numpy_class`` attribute
    specifying the corresponding numpy-based class.

    Examples:
        >>> from anomalib.dataclasses.numpy import NumpyImageItem
        >>> @dataclass
        ... class TorchImageItem(ToNumpyMixin[NumpyImageItem]):
        ...     numpy_class = NumpyImageItem
        ...     image: torch.Tensor
        ...     gt_label: torch.Tensor
        ...
        >>> torch_item = TorchImageItem(
        ...     image=torch.rand(3, 224, 224),
        ...     gt_label=torch.tensor(1)
        ... )
        >>> numpy_item = torch_item.to_numpy()
        >>> isinstance(numpy_item, NumpyImageItem)
        True
    """

    numpy_class: ClassVar[Callable]

    def __init_subclass__(cls, **kwargs) -> None:
        """Ensure that the subclass has the required attributes.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            AttributeError: If the subclass does not define ``numpy_class``.
        """
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "numpy_class"):
            msg = f"{cls.__name__} must have a 'numpy_class' attribute."
            raise AttributeError(msg)

    def to_numpy(self) -> NumpyT:
        """Convert the batch to a NumpyBatch object.

        Returns:
            NumpyT: The converted numpy batch object.
        """
        batch_dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                batch_dict[f.name] = value.detach().cpu().numpy()
            elif hasattr(value, "to_numpy"):
                batch_dict[f.name] = value.to_numpy()
            else:
                batch_dict[f.name] = value
        return self.numpy_class(**batch_dict)


@dataclass
class DatasetItem(Generic[ImageT], _GenericItem[torch.Tensor, ImageT, Mask, str]):
    """Base dataclass for individual items in Anomalib datasets using PyTorch.

    This class extends the generic ``_GenericItem`` class to provide a
    PyTorch-specific implementation for single data items in Anomalib datasets.
    It handles various types of data (e.g., images, labels, masks) represented as
    PyTorch tensors.

    The class uses generic types to allow flexibility in the image representation,
    which can vary depending on the specific use case (e.g., standard images,
    video clips).

    Note:
        This class is typically subclassed to create more specific item types
        (e.g., ``ImageItem``, ``VideoItem``) with additional fields and methods.
    """


@dataclass
class Batch(Generic[ImageT], _GenericBatch[torch.Tensor, ImageT, Mask, list[str]]):
    """Base dataclass for batches of items in Anomalib datasets using PyTorch.

    This class extends the generic ``_GenericBatch`` class to provide a
    PyTorch-specific implementation for batches of data in Anomalib datasets.
    It handles collections of data items (e.g., multiple images, labels, masks)
    represented as PyTorch tensors.

    The class uses generic types to allow flexibility in the image representation,
    which can vary depending on the specific use case (e.g., standard images,
    video clips).

    Note:
        This class is typically subclassed to create more specific batch types
        (e.g., ``ImageBatch``, ``VideoBatch``) with additional fields and methods.
    """

    def keys(self, include_none: bool = True) -> list[str]:
        """Return a list of field names in the Batch.

        Args:
            include_none: If True, returns all possible field names including those
                that are None. If False, returns only field names that have non-None values.
                Defaults to True for backward compatibility.

        Returns:
            List of field names that can be accessed on this Batch instance.
            When include_none=True, includes all fields from the input, output, and any
            additional field classes that the specific batch type inherits from.
            When include_none=False, includes only fields with actual data.

        Example:
            >>> # Using any batch subclass (e.g., ImageBatch)
            >>> batch = Batch(image=torch.rand(2, 3, 224, 224))
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
            >>> # Using any batch subclass (e.g., ImageBatch)
            >>> batch = Batch(image=torch.rand(2, 3, 224, 224))
            >>> batch["image"].shape
            torch.Size([2, 3, 224, 224])
            >>> batch["gt_label"]
            None
        """
        if not hasattr(self, key):
            msg = f"Field '{key}' not found in {self.__class__.__name__}"
            raise KeyError(msg)
        return getattr(self, key)
