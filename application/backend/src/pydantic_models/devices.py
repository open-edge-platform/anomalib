# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel


class DeviceList(BaseModel):
    devices: list[str]

    model_config = {
        "json_schema_extra": {
            "example": {
                "devices": ["CPU", "XPU", "NPU"],
            }
        }
    }
