# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""System information endpoints for feedback and diagnostics."""

import io
import platform
import zipfile
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

from anyio import Path as AsyncPath
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from loguru import logger

from api.endpoints import API_PREFIX
from pydantic_models import AcceleratorInfo, LibraryVersions, SystemInfo
from settings import get_settings

system_info_router = APIRouter(
    prefix=API_PREFIX + "/system",
    tags=["System"],
)
settings = get_settings()


def _get_package_version(package_name: str) -> str:
    """Safely get version of an installed package.

    Args:
        package_name: Name of the package to check. If the package is not installed, return N/A.

    Returns:
        Version string if package is installed
    """
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "N/A"


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
    logs_dir = AsyncPath(settings.log_dir)

    # Create zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Zip the entire logs directory
        async for entity in logs_dir.rglob("*"):
            if await entity.is_file():
                zip_file.write(
                    str(entity),
                    arcname=str(entity.relative_to(logs_dir.parent)),
                )

    # Seek to the beginning of the buffer before returning
    zip_buffer.seek(0)

    # Generate filename with timestamp
    # since this won't run in distributed setting, using local time zone
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"geti_inspect_logs_{timestamp}.zip"

    return StreamingResponse(
        iter([zip_buffer.getvalue()]),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
