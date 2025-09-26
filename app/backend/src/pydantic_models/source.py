# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from os import getenv
from typing import Annotated, Literal
from urllib.parse import urlparse, urlunparse

from pydantic import Field, TypeAdapter

from pydantic_models.base import BaseIDNameModel

IP_CAMERA_USERNAME = "IP_CAMERA_USERNAME"
IP_CAMERA_PASSWORD = "IP_CAMERA_PASSWORD"  # noqa: S105


class SourceType(StrEnum):
    # TODO: remove or refactor "DISCONNECTED" into separate enums if needed
    DISCONNECTED = "disconnected"
    WEBCAM = "webcam"
    IP_CAMERA = "ip_camera"
    VIDEO_FILE = "video_file"
    IMAGES_FOLDER = "images_folder"


class DisconnectedSourceConfig(BaseIDNameModel):
    source_type: Literal[SourceType.DISCONNECTED] = SourceType.DISCONNECTED
    name: str = "No Source"


class WebcamSourceConfig(BaseIDNameModel):
    source_type: Literal[SourceType.WEBCAM]
    device_id: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "webcam",
                "name": "Webcam 0",
                "id": "f9e0ae4f-d96c-4304-baab-2ab845362d03",
                "device_id": 0,
            }
        }
    }


class IPCameraSourceConfig(BaseIDNameModel):
    source_type: Literal[SourceType.IP_CAMERA]
    stream_url: str
    auth_required: bool = False

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "ip_camera",
                "name": "Street Camera 123",
                "id": "3d055c8a-2536-46ea-8f3c-832bd6f8bbdc",
                "stream_url": "http://example.com/stream",
                "auth_required": True,
            }
        }
    }

    def get_configured_stream_url(self) -> str:
        """Configure stream URL with authentication if required."""
        if not self.auth_required:
            return self.stream_url

        username = getenv(IP_CAMERA_USERNAME)
        password = getenv(IP_CAMERA_PASSWORD)

        if not username or not password:
            raise RuntimeError("IP camera credentials not provided.")

        # Modify the stream URL to include authentication
        uri = urlparse(self.stream_url)
        netloc = f"{username}:{password}@{uri.netloc}"
        return urlunparse((uri.scheme, netloc, uri.path, uri.params, uri.query, uri.fragment))


class VideoFileSourceConfig(BaseIDNameModel):
    source_type: Literal[SourceType.VIDEO_FILE]
    video_path: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "video_file",
                "name": "Sample Video",
                "id": "712750b2-5a82-47ee-8fba-f3dc96cb615d",
                "video_path": "/path/to/video.mp4",
            }
        }
    }


class ImagesFolderSourceConfig(BaseIDNameModel):
    source_type: Literal[SourceType.IMAGES_FOLDER]
    images_folder_path: str
    ignore_existing_images: bool

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_type": "images_folder",
                "name": "Best Photos",
                "id": "4a580a0e-b841-4c70-bf88-2d68a28f780d",
                "images_folder_path": "/path/to/images",
                "ignore_existing_images": True,
            }
        }
    }


Source = Annotated[
    WebcamSourceConfig
    | IPCameraSourceConfig
    | VideoFileSourceConfig
    | ImagesFolderSourceConfig
    | DisconnectedSourceConfig,
    Field(discriminator="source_type"),
]

SourceAdapter: TypeAdapter[Source] = TypeAdapter(Source)
