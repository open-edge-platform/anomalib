# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel


class DeviceList(BaseModel):
    devices: list[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "devices": ["CPU", "XPU", "NPU"],
            },
        },
    }


class Camera(BaseModel):
    index: int
    name: str


class CameraList(BaseModel):
    devices: list[Camera]

    model_config = {
        "json_schema_extra": {
            "example": {
                "devices": [
                    {"index": 1200, "name": "camera_name1"},
                    {"index": 1201, "name": "camera_name2"},
                    {"index": 1202, "name": "camera_name3"},
                ],
            },
        },
    }
