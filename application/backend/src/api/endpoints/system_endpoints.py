# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""System information endpoints for feedback and diagnostics."""

import io
import zipfile
from datetime import datetime
from typing import Annotated

from anyio import Path as AsyncPath
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.dependencies.dependencies import get_system_service
from api.endpoints import API_PREFIX
from pydantic_models import SystemInfo
from pydantic_models.system import CameraInfo, DeviceInfo
from services.system_service import SystemService
from settings import get_settings

system_router = APIRouter(
    prefix=API_PREFIX + "/system",
    tags=["System"],
)
settings = get_settings()


@system_router.get("/info")
async def get_system_info(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> SystemInfo:
    """Get system information for feedback and diagnostics.

    Returns:
        SystemInfo containing OS details, app version, library versions,
        and devices info.
    """
    return system_service.get_system_info()


@system_router.get("/devices/inference")
async def get_inference_devices(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> list[DeviceInfo]:
    """Returns the list of available compute devices (CPU, Intel XPU)."""
    return system_service.get_inference_devices()


@system_router.get("/devices/training")
async def get_training_devices(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> list[DeviceInfo]:
    """Returns the list of available training devices (CPU, Intel XPU, NVIDIA CUDA)."""
    return system_service.get_training_devices()


@system_router.get("/devices/camera")
async def get_camera_devices(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> list[CameraInfo]:
    """Returns the list of available camera devices."""
    return system_service.get_camera_devices()


@system_router.get("/metrics/memory")
async def get_memory(
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> dict:
    """Returns the used memory in MB and total available memory in MB."""
    used, total = system_service.get_memory_usage()
    return {"used": int(used), "total": int(total)}


@system_router.post("/logs:export")
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
    # since this won't run in distributed setting, using local timezone
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"geti_inspect_logs_{timestamp}.zip"

    return StreamingResponse(
        iter([zip_buffer.getvalue()]),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@system_router.get("/datapath")
async def get_data_path() -> str:
    """Get the data path used by the application.

    Returns:
        str: The data path.
    """
    return str(get_settings().data_dir)
