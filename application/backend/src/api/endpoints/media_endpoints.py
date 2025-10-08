# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile
from fastapi.responses import FileResponse

from api.dependencies import get_media_id, get_media_service, get_project_id
from api.endpoints import API_PREFIX
from api.media_rest_validator import MediaRestValidator
from pydantic_models import Media, MediaList
from services.media_service import MediaService

logger = logging.getLogger(__name__)

media_api_prefix_url = API_PREFIX + "/projects/{project_id}"
media_router = APIRouter(
    prefix=media_api_prefix_url,
    tags=["media"],
)


@media_router.get("/images")
async def get_media_list(
    media_service: Annotated[MediaService, Depends(get_media_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
) -> MediaList:
    """Endpoint to get list of all media"""
    return await media_service.get_media_list(project_id=project_id)


@media_router.get("/images/{media_id}/full", response_model_exclude_none=True)
async def get_media(
    media_service: Annotated[MediaService, Depends(get_media_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
    media_id: Annotated[UUID, Depends(get_media_id)],
) -> FileResponse:
    """Endpoint to get media metadata by ID"""
    return FileResponse(await media_service.get_media_file_path(project_id=project_id, media_id=media_id))


@media_router.get("/images/{media_id}/thumbnail", response_model_exclude_none=True)
async def get_media_thumbnail(
    media_service: Annotated[MediaService, Depends(get_media_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
    media_id: Annotated[UUID, Depends(get_media_id)],
) -> FileResponse:
    """Return a PNG thumbnail for the requested image."""
    return FileResponse(await media_service.get_thumbnail_file_path(project_id=project_id, media_id=media_id))


@media_router.post("/capture", response_model_exclude_none=True)
async def capture_image(
    media_service: Annotated[MediaService, Depends(get_media_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
    file: Annotated[UploadFile, Depends(MediaRestValidator.validate_image_file)],
    background_tasks: BackgroundTasks,
) -> Media:
    """Endpoint to capture an image"""
    image_bytes = await file.read()
    media = await media_service.upload_image(
        project_id=project_id, file=file, image_bytes=image_bytes, is_anomalous=False
    )

    background_tasks.add_task(
        media_service.generate_thumbnail,
        project_id=project_id,
        media_id=media.id,
        image_bytes=image_bytes,
    )
    return media
