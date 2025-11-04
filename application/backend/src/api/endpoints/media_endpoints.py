# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from api.dependencies import get_media_id, get_media_service, get_project_id
from api.endpoints import API_PREFIX
from api.media_rest_validator import MediaRestValidator
from pydantic_models import Media, MediaList
from services.media_service import MediaService

media_api_prefix_url = API_PREFIX + "/projects/{project_id}"
media_router = APIRouter(
    prefix=media_api_prefix_url,
    tags=["media"],
)


@media_router.get(
    "/images",
    responses={
        status.HTTP_200_OK: {"description": "List of media items retrieved successfully"},
        status.HTTP_404_NOT_FOUND: {"description": "Project not found"},
    },
)
async def get_media_list(
    media_service: Annotated[MediaService, Depends(get_media_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
) -> MediaList:
    """Endpoint to get list of all media"""
    return await media_service.get_media_list(project_id=project_id)


@media_router.get(
    "/images/{media_id}/full",
    response_model_exclude_none=True,
    responses={
        status.HTTP_200_OK: {"description": "Media file retrieved successfully"},
        status.HTTP_404_NOT_FOUND: {"description": "Media not found"},
    },
)
async def get_media(
    media_service: Annotated[MediaService, Depends(get_media_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
    media_id: Annotated[UUID, Depends(get_media_id)],
) -> FileResponse:
    """Endpoint to get media metadata by ID"""
    try:
        return FileResponse(await media_service.get_media_file_path(project_id=project_id, media_id=media_id))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Media with ID {media_id} not found",
        ) from e


@media_router.get(
    "/images/{media_id}/thumbnail",
    response_model_exclude_none=True,
    responses={
        status.HTTP_200_OK: {"description": "Thumbnail retrieved successfully"},
        status.HTTP_404_NOT_FOUND: {"description": "Media not found"},
    },
)
async def get_media_thumbnail(
    media_service: Annotated[MediaService, Depends(get_media_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
    media_id: Annotated[UUID, Depends(get_media_id)],
) -> FileResponse:
    """Return a PNG thumbnail for the requested image."""
    try:
        return FileResponse(await media_service.get_thumbnail_file_path(project_id=project_id, media_id=media_id))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Media with ID {media_id} not found",
        ) from e


@media_router.delete(
    "/images/{media_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        status.HTTP_204_NO_CONTENT: {
            "description": "Image successfully deleted",
        },
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid image ID"},
        status.HTTP_404_NOT_FOUND: {"description": "Image not found"},
    },
)
async def delete_media(
    media_service: Annotated[MediaService, Depends(get_media_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
    media_id: Annotated[UUID, Depends(get_media_id)],
) -> None:
    """Remove an image"""
    await media_service.delete_media(media_id=media_id, project_id=project_id)


@media_router.post(
    "/capture",
    response_model_exclude_none=True,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_201_CREATED: {"description": "Image captured successfully"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid image file"},
    },
)
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
