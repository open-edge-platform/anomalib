# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum, auto

from pydantic import BaseModel, Field


class DeviceType(StrEnum):
    """Enumeration of device types"""

    CPU = auto()
    XPU = auto()
    CUDA = auto()
    MPS = auto()
    NPU = auto()


class DeviceInfo(BaseModel):
    """Device information schema"""

    type: DeviceType = Field(..., description="Device type (cpu, xpu, cuda, mps, npu)")
    name: str = Field(..., description="Device name")
    memory: int | None = Field(None, description="Total memory available to the device, in bytes (null for CPU/NPU)")
    index: int | None = Field(None, description="Device index among those of the same type (null for CPU/NPU)")
    openvino_name: str | None = Field(None, description="Name of the OpenVINO device (inference only)")


class CameraInfo(BaseModel):
    """Camera information schema"""

    index: int = Field(..., description="Camera device index")
    name: str = Field(..., description="Camera device name")


class LibraryVersions(BaseModel):
    """Version information for libraries."""

    anomalib: str | None = None
    python: str
    openvino: str | None = None
    pytorch: str | None = None
    lightning: str | None = None
    torchmetrics: str | None = None
    onnx: str | None = None
    cuda: str | None = None
    cudnn: str | None = None
    xpu_driver: str | None = None


class SystemInfo(BaseModel):
    """System information for feedback and diagnostics."""

    os_name: str
    os_version: str
    platform: str
    app_version: str
    libraries: LibraryVersions
    devices: list[DeviceInfo]

    model_config = {
        "json_schema_extra": {
            "example": {
                "os_name": "Linux",
                "os_version": "5.15.0-generic",
                "platform": "Linux-5.15.0-generic-x86_64-with-glibc2.35",
                "app_version": "0.1.0",
                "libraries": {
                    "anomalib": "2.0.0",
                    "python": "3.11.0",
                    "openvino": "2025.0.0",
                    "pytorch": "2.1.0",
                    "lightning": "2.0.0",
                    "torchmetrics": "1.0.0",
                    "onnx": "1.15.0",
                    "cuda": "12.1",
                    "cudnn": "8.9.0",
                    "xpu_driver": None,
                },
                "devices": [
                    {
                        "type": "CPU",
                        "name": "Intel(R) Core(TM) i7-14900K CPU @ 5.80GHz",
                        "memory": None,
                        "index": None,
                    },
                    {
                        "type": "CUDA",
                        "name": "NVIDIA GeForce RTX 5090",
                        "memory": 34359738368,
                        "index": 0,
                    },
                    {
                        "type": "XPU",
                        "name": "Intel(R) UHD Graphics 630",
                        "memory": 4294967296,
                        "index": 0,
                    },
                ],
            },
        },
    }
