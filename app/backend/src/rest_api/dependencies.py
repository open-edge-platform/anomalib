from functools import lru_cache
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio.session import AsyncSession

from db import get_async_db_session
from services import JobService, MediaService, ProjectService
from services.model_service import ModelService


@lru_cache
def get_project_service(db_session: Annotated[AsyncSession, Depends(get_async_db_session)]) -> ProjectService:
    """Provides a ProjectService"""
    return ProjectService(db_session=db_session)


@lru_cache
def get_job_service(db_session: Annotated[AsyncSession, Depends(get_async_db_session)]) -> JobService:
    """Provides a JobService"""
    return JobService(db_session=db_session)


@lru_cache
def get_media_service(db_session: Annotated[AsyncSession, Depends(get_async_db_session)]) -> MediaService:
    """Provides a MediaService"""
    return MediaService(db_session=db_session)


@lru_cache
def get_model_service(db_session: Annotated[AsyncSession, Depends(get_async_db_session)]) -> ModelService:
    """Provides a MediaService"""
    return ModelService(db_session=db_session)


def is_valid_uuid(identifier: str) -> bool:
    """
    Check if a given string identifier is formatted as a valid UUID

    :param identifier: String to check
    :return: True if valid UUID, False otherwise
    """
    try:
        UUID(identifier)
    except ValueError:
        return False
    return True


def get_uuid(identifier: str, name: str = "DIO") -> UUID:
    """Initializes and validates a source ID"""
    if not is_valid_uuid(identifier):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid {name} ID")
    return UUID(identifier)


def get_project_id(project_id: str) -> UUID:
    """Initializes and validates a project ID"""
    return get_uuid(project_id, "project")


def get_media_id(media_id: str) -> UUID:
    """Initializes and validates a media ID"""
    return get_uuid(media_id, "media")


def get_model_id(model_id: str) -> UUID:
    """Initializes and validates a media ID"""
    return get_uuid(model_id, "model")
