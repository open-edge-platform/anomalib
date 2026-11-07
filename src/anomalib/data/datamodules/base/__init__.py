# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base Anomalib Data Modules."""

from .image import AnomalibDataModule
from .video import AnomalibVideoDataModule

__all__ = ["AnomalibDataModule", "AnomalibVideoDataModule"]
