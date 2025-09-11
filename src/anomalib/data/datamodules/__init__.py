# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomalib Data Modules."""

from .depth import Folder3D, MVTec3D
from .image import BMAD, MPDD, VAD, BTech, Datumaro, Folder, Kolektor, MVTec, MVTecAD, Tabular, Visa
from .video import Avenue, ShanghaiTech, UCSDped

__all__ = [
    "Folder3D",
    "MVTec3D",
    "BTech",
    "Datumaro",
    "Folder",
    "Kolektor",
    "MPDD",
    "MVTec",  # Include MVTec for backward compatibility
    "MVTecAD",
    "Tabular",
    "VAD",
    "Visa",
    "Avenue",
    "ShanghaiTech",
    "UCSDped",
    "BMAD",
]
