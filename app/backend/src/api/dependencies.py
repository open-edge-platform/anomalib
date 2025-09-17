from functools import lru_cache
from uuid import UUID

from fastapi import HTTPException, status

from services import ActivePipelineService, JobService, MediaService, ProjectService
from services.model_service import ModelService


@lru_cache
def get_project_service() -> ProjectService:
    """Provides a ProjectService"""
    return ProjectService()


@lru_cache
def get_job_service() -> JobService:
    """Provides a JobService"""
    return JobService()


@lru_cache
def get_media_service() -> MediaService:
    """Provides a MediaService"""
    return MediaService()


@lru_cache
def get_model_service() -> ModelService:
    """Provides a MediaService"""
    return ModelService()


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


@lru_cache
def get_active_pipeline_service() -> ActivePipelineService:
    """Provides an ActivePipelineService instance for managing the active pipeline."""
    return ActivePipelineService()
