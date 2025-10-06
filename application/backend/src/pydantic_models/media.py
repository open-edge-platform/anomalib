# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from pydantic_models.base import BaseIDModel


class Subset(StrEnum):
    TRAINING = "training"
    TESTING = "testing"


class ImageExtension(StrEnum):
    JPG = ".jpg"
    JPEG = ".jpeg"
    PNG = ".png"
    BMP = ".bmp"
    TIF = ".tif"
    TIFF = ".tiff"
    JFIF = ".jfif"
    WEBP = ".webp"


class Media(BaseIDModel):
    project_id: UUID
    filename: str
    size: int = Field(..., ge=0)
    is_anomalous: bool
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    subset: Subset | None = None
    created_at: datetime | None = None


class MediaList(BaseModel):
    media: list[Media]
