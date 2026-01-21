# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel


class LibraryVersions(BaseModel):
    """Version information for ML/DL libraries."""

    python: str
    pytorch: str | None = None
    lightning: str | None = None
    torchmetrics: str | None = None
    openvino: str | None = None
    onnx: str | None = None
    anomalib: str | None = None


class AcceleratorInfo(BaseModel):
    """Information about hardware accelerators."""

    cuda_available: bool = False
    cuda_version: str | None = None
    cudnn_version: str | None = None
    gpu_name: str | None = None
    xpu_available: bool = False
    xpu_version: str | None = None
    xpu_name: str | None = None


class SystemInfo(BaseModel):
    """System information for feedback and diagnostics."""

    os_name: str
    os_version: str
    platform: str
    app_version: str
    libraries: LibraryVersions
    accelerators: AcceleratorInfo

    model_config = {
        "json_schema_extra": {
            "example": {
                "os_name": "Linux",
                "os_version": "5.15.0-generic",
                "platform": "Linux-5.15.0-generic-x86_64-with-glibc2.35",
                "app_version": "0.1.0",
                "libraries": {
                    "python": "3.11.0",
                    "pytorch": "2.1.0",
                    "lightning": "2.0.0",
                    "torchmetrics": "1.0.0",
                    "openvino": "2024.0.0",
                    "onnx": "1.15.0",
                    "anomalib": "1.0.0",
                },
                "accelerators": {
                    "cuda_available": True,
                    "cuda_version": "12.1",
                    "cudnn_version": "8.9.0",
                    "gpu_name": "NVIDIA GeForce RTX 3080",
                    "xpu_available": False,
                    "xpu_version": None,
                    "xpu_name": None,
                },
            },
        },
    }
