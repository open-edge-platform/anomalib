# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""System information endpoints for feedback and diagnostics."""

import io
import platform
import zipfile
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from api.endpoints import API_PREFIX
from settings import get_settings

system_info_router = APIRouter(
    prefix=API_PREFIX + "/system",
    tags=["System"],
)

settings = get_settings()


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


class LogFile(BaseModel):
    """A single log file's contents."""

    name: str
    content: str


class SystemInfo(BaseModel):
    """System information for feedback and diagnostics."""

    os_name: str
    os_version: str
    platform: str
    app_version: str
    app_name: str
    libraries: LibraryVersions
    accelerators: AcceleratorInfo

    model_config = {
        "json_schema_extra": {
            "example": {
                "os_name": "Linux",
                "os_version": "5.15.0-generic",
                "platform": "Linux-5.15.0-generic-x86_64-with-glibc2.35",
                "app_version": "0.1.0",
                "app_name": "Geti Inspect",
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
            }
        }
    }


def _get_package_version(package_name: str) -> str:
    """Safely get version of an installed package.

    Args:
        package_name: Name of the package to check.

    Returns:
        Version string if package is installed
    """
    return version(package_name)


def _get_library_versions() -> LibraryVersions:
    """Collect versions of ML/DL libraries.

    Returns:
        LibraryVersions with version info for each library.
    """
    return LibraryVersions(
        python=platform.python_version(),
        pytorch=_get_package_version("torch"),
        lightning=_get_package_version("lightning"),
        torchmetrics=_get_package_version("torchmetrics"),
        openvino=_get_package_version("openvino"),
        onnx=_get_package_version("onnx"),
        anomalib=_get_package_version("anomalib"),
    )


def _get_accelerator_info() -> AcceleratorInfo:
    """Collect information about available hardware accelerators.

    Returns:
        AcceleratorInfo with CUDA and XPU availability and versions.
    """
    info = AcceleratorInfo()

    # Check CUDA availability and version
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        info.cuda_available = True
        # Get CUDA version from torch
        info.cuda_version = torch.version.cuda
        # Get GPU name
        try:
            info.gpu_name = torch.cuda.get_device_name(0)
        except Exception as exception:
            logger.info(f"Error getting CUDA device name: {exception}")
        # Get cuDNN version
        if torch.backends.cudnn.is_available():
            cudnn_ver = torch.backends.cudnn.version()
            if cudnn_ver:
                # Format: major * 1000 + minor * 100 + patch
                major = cudnn_ver // 1000
                minor = (cudnn_ver % 1000) // 100
                patch = cudnn_ver % 100
                info.cudnn_version = f"{major}.{minor}.{patch}"

    # Check XPU (Intel GPU) availability and version

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        info.xpu_available = True
        # Get device properties using torch.xpu.get_device_properties
        try:
            device_props = torch.xpu.get_device_properties(torch.xpu.current_device())
            info.xpu_name = device_props.name
            info.xpu_version = getattr(device_props, "driver_version", "N/A")
        except Exception as exception:
            logger.info(f"Error getting XPU device properties: {exception}. Setting XPU version to N/A.")
            info.xpu_version = "N/A"

    return info


def _get_logs() -> list[LogFile]:
    """Get recent log entries from all application log files.


    Returns:
        List of LogFile objects containing log file names and their contents.
    """
    logs_dir = Path(settings.log_dir)
    log_files = [
        "app.log",
        "training.log",
        "inference.log",
        "dispatching.log",
        "stream_loader.log",
    ]

    result = []
    for log_name in log_files:
        log_path = logs_dir / log_name
        with open(log_path, encoding="utf-8") as f:
            content = f.read()
            if content:
                result.append(LogFile(name=log_name, content=content))

    return result


@system_info_router.get("/info")
async def get_system_info() -> SystemInfo:
    """Get system information for feedback and diagnostics.

    Returns:
        SystemInfo containing OS details, app version, library versions,
        and accelerator info.
    """

    return SystemInfo(
        os_name=platform.system(),
        os_version=platform.release(),
        platform=platform.platform(),
        app_version=settings.version,
        app_name=settings.app_name,
        libraries=_get_library_versions(),
        accelerators=_get_accelerator_info(),
    )


@system_info_router.get("/logs")
async def download_logs() -> StreamingResponse:
    """Download application logs as a zip file.

    Returns:
        StreamingResponse containing a zip file with all available logs.
    """
    logs = _get_logs()

    # Create zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for log in logs:
            zip_file.writestr(log.name, log.content)

    zip_buffer.seek(0)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"geti_inspect_logs_{timestamp}.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
