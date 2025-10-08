# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch Dataset implementations for anomaly detection in images.

This module provides dataset implementations for various image anomaly detection
datasets:

- ``BMADDataset``: BMAD dataset containing medical images
- ``BTechDataset``: BTech dataset containing industrial objects
- ``DatumaroDataset``: Dataset in Datumaro format (Intel Geti™ export)
- ``FolderDataset``: Custom dataset from folder structure
- ``KolektorDataset``: Kolektor surface defect dataset
- ``MVTecADDataset``: MVTec AD dataset with industrial objects
- ``MVTecLOCODataset``: MVTec LOCO dataset with logical and structural anomalies
- ``TabularDataset``: Custom tabular dataset with image paths and labels
- ``VAD``: Valeo Anomaly Detection Dataset
- ``VisaDataset``: Visual Anomaly dataset

Example:
    >>> from anomalib.data.datasets import MVTecADDataset
    >>> dataset = MVTecADDataset(
    ...     root="./datasets/MVTec",
    ...     category="bottle",
    ...     split="train"
    ... )
"""

from .bmad import BMADDataset
from .btech import BTechDataset
from .datumaro import DatumaroDataset
from .folder import FolderDataset
from .kolektor import KolektorDataset
from .mpdd import MPDDDataset
from .mvtec_loco import MVTecLOCODataset
from .mvtecad import MVTecADDataset, MVTecDataset
from .mvtecad2 import MVTecAD2Dataset
from .realiad import RealIADDataset
from .tabular import TabularDataset
from .vad import VADDataset
from .visa import VisaDataset

__all__ = [
    "BMADDataset",
    "BTechDataset",
    "DatumaroDataset",
    "FolderDataset",
    "KolektorDataset",
    "MPDDDataset",
    "MVTecDataset",
    "MVTecADDataset",
    "MVTecAD2Dataset",
    "MVTecLOCODataset",
    "RealIADDataset",
    "TabularDataset",
    "VADDataset",
    "VisaDataset",
]
