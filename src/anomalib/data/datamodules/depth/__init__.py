# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomalib Depth Data Modules."""

from enum import Enum

from .adam_3d import ADAM3D
from .folder_3d import Folder3D
from .mvtec_3d import MVTec3D


class DepthDataFormat(str, Enum):
    """Supported Depth Dataset Types."""

    MVTEC_3D = "mvtec_3d"
    FOLDER_3D = "folder_3d"
    ADAM_3D = "adam_3d"


__all__ = ["Folder3D", "MVTec3D", "ADAM3D"]
