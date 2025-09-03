# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import FileResponse

from models import Media
from rest_api.dependencies import get_media_service, get_project_id, get_media_id
from rest_api.endpoints import API_PREFIX
from rest_api.media_rest_validator import MediaRestValidator
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
    project_id: Annotated[UUID, Depends(get_project_id)]
):
    return await media_service.get_media_list(project_id=project_id)


@media_router.get("/images/{media_id}", response_model=Media, response_model_exclude_none=True)
async def get_media(
    media_service: Annotated[MediaService, Depends(get_media_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
    media_id: Annotated[UUID, Depends(get_media_id)]
):
    return FileResponse(await media_service.get_media_file_path(project_id=project_id, media_id=media_id))


@media_router.post("/capture", response_model=Media, response_model_exclude_none=True)
async def capture_image(
    media_service: Annotated[MediaService, Depends(get_media_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
    file: UploadFile = Depends(MediaRestValidator.validate_image_file),
):
    return await media_service.upload_image(project_id=project_id, file=file, is_anomalous=False)
