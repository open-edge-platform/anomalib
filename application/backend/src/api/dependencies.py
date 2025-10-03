from functools import lru_cache
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status

from core import Scheduler
from services import (
    ActivePipelineService,
    ConfigurationService,
    JobService,
    MediaService,
    PipelineService,
    ProjectService,
)
from services.metrics_service import MetricsService
from services.model_service import ModelService
from webrtc.manager import WebRTCManager


async def get_project_service() -> ProjectService:
    """Provides a ProjectService"""
    return ProjectService()


async def get_job_service() -> JobService:
    """Provides a JobService"""
    return JobService()


async def get_media_service() -> MediaService:
    """Provides a MediaService"""
    return MediaService()


async def get_scheduler(request: Request) -> Scheduler:
    """Provides the global Scheduler instance."""
    return request.app.state.scheduler


async def get_model_service(scheduler: Annotated[Scheduler, Depends(get_scheduler)]) -> ModelService:
    """Provides a ModelService with access to the model reload event"""
    return ModelService(mp_model_reload_event=scheduler.mp_model_reload_event)


@lru_cache
def get_metrics_service(scheduler: Annotated[Scheduler, Depends(get_scheduler)]) -> MetricsService:
    """Provides a MetricsService instance for collecting and retrieving metrics."""
    return MetricsService(scheduler.shm_metrics.name, scheduler.shm_metrics_lock)


async def get_active_pipeline_service(scheduler: Annotated[Scheduler, Depends(get_scheduler)]) -> ActivePipelineService:
    """
    Provides an ActivePipelineService instance for managing the active pipeline.

    This dependency is designed for use in FastAPI endpoints and creates a service
    without the daemon thread. For worker processes, create ActivePipelineService
    directly with the config_changed_condition.
    """
    # Create service without daemon thread for API endpoints
    return await ActivePipelineService.create(scheduler.mp_config_changed_condition)


@lru_cache
def get_configuration_service(
    active_pipeline_service: Annotated[ActivePipelineService, Depends(get_active_pipeline_service)],
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
) -> ConfigurationService:
    """Provides a ConfigurationService instance with the active pipeline service and config changed condition."""
    return ConfigurationService(
        active_pipeline_service=active_pipeline_service,
        config_changed_condition=scheduler.mp_config_changed_condition,
    )


@lru_cache
def get_pipeline_service(
    active_pipeline_service: Annotated[ActivePipelineService, Depends(get_active_pipeline_service)],
    metrics_service: Annotated[MetricsService, Depends(get_metrics_service)],
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
    model_service: Annotated[ModelService, Depends(get_model_service)],
) -> PipelineService:
    """Provides a PipelineService instance with the active pipeline service and config changed condition."""
    return PipelineService(
        active_pipeline_service=active_pipeline_service,
        metrics_service=metrics_service,
        config_changed_condition=scheduler.mp_config_changed_condition,
        model_service=model_service,
    )


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


def get_source_id(source_id: str) -> UUID:
    """Initializes and validates a source ID"""
    if not is_valid_uuid(source_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid source ID")
    return UUID(source_id)


def get_sink_id(sink_id: str) -> UUID:
    """Initializes and validates a sink ID"""
    if not is_valid_uuid(sink_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid sink ID")
    return UUID(sink_id)


def get_project_id(project_id: str) -> UUID:
    """Initializes and validates a project ID"""
    return get_uuid(project_id, "project")


def get_media_id(media_id: str) -> UUID:
    """Initializes and validates a media ID"""
    return get_uuid(media_id, "media")


def get_model_id(model_id: str) -> UUID:
    """Initializes and validates a media ID"""
    return get_uuid(model_id, "model")


async def get_webrtc_manager(request: Request) -> WebRTCManager:
    """Provides the global WebRTCManager instance from FastAPI application's state."""
    return request.app.state.webrtc_manager
