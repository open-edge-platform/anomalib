# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum, auto

from pydantic import BaseModel, Field


class DeviceType(StrEnum):
    """Supported compute device types."""

    CPU = auto()
    XPU = auto()
    CUDA = auto()
    MPS = auto()
    NPU = auto()


class DeviceInfo(BaseModel):
    """Compute device information."""

    type: DeviceType = Field(..., description="Device type (cpu, xpu, cuda, mps, npu)")
    name: str = Field(..., description="Device name")
    memory: int | None = Field(None, description="Total memory in bytes (null for CPU/NPU)")
    index: int | None = Field(None, description="Device index among those of the same type (null for CPU/NPU)")
    openvino_name: str | None = Field(None, description="OpenVINO device identifier (inference only)")


class CameraInfo(BaseModel):
    """Camera device information."""

    index: int = Field(..., description="Camera device index")
    name: str = Field(..., description="Camera device name")


class LibraryVersions(BaseModel):
    """Installed library versions for diagnostics."""

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
    is_desktop: bool = Field(..., description="True when running as a packaged desktop (Tauri) application")
    libraries: LibraryVersions
    devices: list[DeviceInfo]

    model_config = {
        "json_schema_extra": {
            "example": {
                "os_name": "Linux",
                "os_version": "5.15.0-generic",
                "platform": "Linux-5.15.0-generic-x86_64-with-glibc2.35",
                "app_version": "0.1.0",
                "is_desktop": False,
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


class LicenseInfo(BaseModel):
    """License details shown in the desktop acceptance dialog.

    The desktop application is distributed under ISSL while the source
    code remains Apache-2.0.  Some bundled components carry their own
    open-source licenses listed in the third-party notices file.
    """

    distribution_license_name: str = Field(..., description="Name of the distribution license (ISSL)")
    distribution_license_url: str = Field(..., description="URL to the full distribution license text")
    source_license_name: str = Field(..., description="Name of the source code license (Apache-2.0)")
    source_license_url: str = Field(..., description="URL to the full source code license text")
    third_party_notices_url: str = Field(..., description="URL to third-party program notices")


class LicenseStatus(BaseModel):
    """License acceptance status returned by ``GET /api/system/license``.

    Desktop builds must show the license dialog on first launch.  All
    other deployments are pre-accepted and ``license`` is ``None``.
    """

    accepted: bool = Field(..., description="Whether the license has been accepted")
    app_version: str = Field(..., description="Current application version")
    is_desktop: bool = Field(..., description="True when running as a packaged desktop application")
    license: LicenseInfo | None = Field(None, description="License details (present only for desktop builds)")


class LicenseAcceptanceResponse(BaseModel):
    """Response returned by ``POST /api/system/license:accept``."""

    accepted: bool = Field(..., description="Whether acceptance was stored successfully")
