# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomalib Datasets.

This module provides datasets and data modules for anomaly detection tasks.

The module contains:
    - Data classes for representing different types of data (images, videos, etc.)
    - Dataset classes for loading and processing data
    - Data modules for use with PyTorch Lightning
    - Helper functions for data loading and validation

Example:
    >>> from anomalib.data import MVTecAD
    >>> datamodule = MVTecAD(
    ...     root="./datasets/MVTecAD",
    ...     category="bottle",
    ...     image_size=(256, 256)
    ... )
"""

from __future__ import annotations

import importlib
import logging

# Dataclasses are lightweight and imported eagerly because they are used
# pervasively throughout the codebase at module level (e.g. type hints in
# ``anomalib.models`` and ``anomalib.data.utils``).  Keeping them eager avoids
# circular-import issues that arise when sub-modules such as
# ``anomalib.data.utils.video`` try to import ``VideoItem`` while
# ``anomalib.data`` is still being initialised.
from .dataclasses import (
    Batch,
    DatasetItem,
    DepthBatch,
    DepthItem,
    ImageBatch,
    ImageItem,
    InferenceBatch,
    NumpyImageBatch,
    NumpyImageItem,
    NumpyVideoBatch,
    NumpyVideoItem,
    VideoBatch,
    VideoItem,
)

# AnomalibDataModule is the base class used for ``__subclasses__()`` discovery
# and by ``get_datamodule()``.  It is intentionally imported eagerly.
from .datamodules.base import AnomalibDataModule

# ---------------------------------------------------------------------------
# Everything below is *lazy* — concrete datamodules, datasets, data-format
# enums and the ``PredictDataset`` helper are only imported on first access.
# ---------------------------------------------------------------------------

_LAZY_IMPORTS: dict[str, str] = {
    # Datamodules - depth
    "ADAM3D": ".datamodules.depth",
    "DepthDataFormat": ".datamodules.depth",
    "Folder3D": ".datamodules.depth",
    "MVTec3D": ".datamodules.depth",
    # Datamodules - image
    "BMAD": ".datamodules.image",
    "MPDD": ".datamodules.image",
    "VAD": ".datamodules.image",
    "BTech": ".datamodules.image",
    "Datumaro": ".datamodules.image",
    "Folder": ".datamodules.image",
    "ImageDataFormat": ".datamodules.image",
    "Kaputt": ".datamodules.image",
    "Kolektor": ".datamodules.image",
    "MVTec": ".datamodules.image",
    "MVTecAD": ".datamodules.image",
    "MVTecAD2": ".datamodules.image",
    "MVTecLOCO": ".datamodules.image",
    "RealIAD": ".datamodules.image",
    "Tabular": ".datamodules.image",
    "Visa": ".datamodules.image",
    # Datamodules - video
    "Avenue": ".datamodules.video",
    "ShanghaiTech": ".datamodules.video",
    "UCSDped": ".datamodules.video",
    "VideoDataFormat": ".datamodules.video",
    # Datasets
    "AnomalibDataset": ".datasets",
    "ADAM3DDataset": ".datasets.depth",
    "Folder3DDataset": ".datasets.depth",
    "MVTec3DDataset": ".datasets.depth",
    "BMADDataset": ".datasets.image",
    "BTechDataset": ".datasets.image",
    "DatumaroDataset": ".datasets.image",
    "FolderDataset": ".datasets.image",
    "KaputtDataset": ".datasets.image",
    "KolektorDataset": ".datasets.image",
    "MPDDDataset": ".datasets.image",
    "MVTecADDataset": ".datasets.image",
    "MVTecLOCODataset": ".datasets.image",
    "RealIADDataset": ".datasets.image",
    "TabularDataset": ".datasets.image",
    "VADDataset": ".datasets.image",
    "VisaDataset": ".datasets.image",
    "AvenueDataset": ".datasets.video",
    "ShanghaiTechDataset": ".datasets.video",
    "UCSDpedDataset": ".datasets.video",
    "PredictDataset": ".predict",
}

logger = logging.getLogger(__name__)


class UnknownDatamoduleError(ModuleNotFoundError):
    """Raised when a datamodule cannot be found."""


def __getattr__(name: str) -> object:
    if name == "DataFormat":
        from enum import Enum
        from itertools import chain

        DepthDataFormat = _get(".datamodules.depth", "DepthDataFormat")
        ImageDataFormat = _get(".datamodules.image", "ImageDataFormat")
        VideoDataFormat = _get(".datamodules.video", "VideoDataFormat")
        DataFormat = Enum(  # type: ignore[misc]
            "DataFormat",
            {i.name: i.value for i in chain(DepthDataFormat, ImageDataFormat, VideoDataFormat)},
        )
        globals()["DataFormat"] = DataFormat
        return DataFormat

    if name in _LAZY_IMPORTS:
        obj = _get(_LAZY_IMPORTS[name], name)
        globals()[name] = obj
        return obj

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def _get(module_path: str, attr: str) -> object:
    mod = importlib.import_module(module_path, __name__)
    return getattr(mod, attr)


def get_datamodule(config) -> AnomalibDataModule:
    """Get Anomaly Datamodule from config.

    Args:
        config: Configuration for the anomaly model. Can be either:
            - DictConfig from OmegaConf
            - ListConfig from OmegaConf
            - Python dictionary

    Returns:
        PyTorch Lightning DataModule configured according to the input.

    Raises:
        UnknownDatamoduleError: If the specified datamodule cannot be found.

    Example:
        >>> from omegaconf import DictConfig
        >>> config = DictConfig({
        ...     "data": {
        ...         "class_path": "MVTecAD",
        ...         "init_args": {"root": "./datasets/MVTec"}
        ...     }
        ... })
        >>> datamodule = get_datamodule(config)
    """
    from omegaconf import DictConfig

    from anomalib.utils.config import to_tuple

    logger.info("Loading the datamodule and dataset class from the config.")

    if isinstance(config, dict):
        config = DictConfig(config)
    config_ = config.data if "data" in config else config

    module = importlib.import_module("anomalib.data")
    data_class_name = config_.class_path.split(".")[-1]
    if not hasattr(module, data_class_name):
        logger.error(
            f"Dataclass '{data_class_name}' not found in module '{module.__name__}'. "
            f"Available classes are {AnomalibDataModule.__subclasses__()}",
        )
        error_str = f"Dataclass '{data_class_name}' not found in module '{module.__name__}'."
        raise UnknownDatamoduleError(error_str)
    dataclass = getattr(module, data_class_name)

    init_args = {**config_.get("init_args", {})}
    if "image_size" in init_args:
        init_args["image_size"] = to_tuple(init_args["image_size"])
    return dataclass(**init_args)


__all__ = [
    # Base Classes
    "AnomalibDataModule",
    "AnomalibDataset",
    # Data Classes
    "Batch",
    "DatasetItem",
    "DepthBatch",
    "DepthItem",
    "ImageBatch",
    "ImageItem",
    "InferenceBatch",
    "NumpyImageBatch",
    "NumpyImageItem",
    "NumpyVideoBatch",
    "NumpyVideoItem",
    "VideoBatch",
    "VideoItem",
    # Data Formats
    "DataFormat",
    "DepthDataFormat",
    "ImageDataFormat",
    "VideoDataFormat",
    # Depth Data Modules
    "Folder3D",
    "MVTec3D",
    "ADAM3D",
    # Image Data Modules
    "BMAD",
    "BTech",
    "Datumaro",
    "Folder",
    "Kaputt",
    "Kolektor",
    "MPDD",
    "MVTec",
    "MVTecAD",
    "MVTecAD2",
    "MVTecLOCO",
    "RealIAD",
    "Tabular",
    "VAD",
    "Visa",
    # Video Data Modules
    "Avenue",
    "ShanghaiTech",
    "UCSDped",
    # Datasets
    "Folder3DDataset",
    "MVTec3DDataset",
    "BTechDataset",
    "DatumaroDataset",
    "FolderDataset",
    "KaputtDataset",
    "KolektorDataset",
    "MPDDDataset",
    "ADAM3DDataset",
    "MVTecADDataset",
    "MVTecLOCODataset",
    "TabularDataset",
    "VADDataset",
    "VisaDataset",
    "AvenueDataset",
    "ShanghaiTechDataset",
    "UCSDpedDataset",
    "PredictDataset",
    "BMADDataset",
    "RealIADDataset",
    # Functions
    "get_datamodule",
    # Exceptions
    "UnknownDatamoduleError",
]
