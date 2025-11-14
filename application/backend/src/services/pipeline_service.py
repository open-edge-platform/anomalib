# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from multiprocessing.synchronize import Condition
from uuid import UUID

from sqlalchemy.ext.asyncio.session import AsyncSession

from db import get_async_db_session_ctx
from exceptions import ResourceNotFoundException
from pydantic_models import Pipeline
from repositories import PipelineRepository
from services import ActivePipelineService
from services.model_service import ModelService

MSG_ERR_DELETE_RUNNING_PIPELINE = "Cannot delete a running pipeline."


class PipelineService:
    def __init__(
        self,
        active_pipeline_service: ActivePipelineService,
        config_changed_condition: Condition,
        model_service: ModelService,
    ) -> None:
        self._active_pipeline_service: ActivePipelineService = active_pipeline_service
        self._config_changed_condition: Condition = config_changed_condition
        self._model_service: ModelService = model_service

    def _notify_source_changed(self) -> None:
        with self._config_changed_condition:
            self._config_changed_condition.notify_all()

    async def _notify_sink_changed(self) -> None:
        await self._active_pipeline_service.reload()

    async def _notify_pipeline_changed(self) -> None:
        self._notify_source_changed()
        await self._notify_sink_changed()

    @staticmethod
    async def get_pipeline_by_id(project_id: UUID, session: AsyncSession | None = None) -> Pipeline:
        """Retrieve a pipeline by project ID."""
        if session is None:
            async with get_async_db_session_ctx() as db_session:
                repo = PipelineRepository(db_session)
                pipeline = await repo.get_by_id(project_id)
        else:
            repo = PipelineRepository(session)
            pipeline = await repo.get_by_id(project_id)
        if not pipeline:
            raise ResourceNotFoundException(resource_id=project_id, resource_name="pipeline")
        return pipeline

    async def update_pipeline(self, project_id: UUID, partial_config: dict) -> Pipeline:
        """Update an existing pipeline."""
        async with get_async_db_session_ctx() as session:
            pipeline = await self.get_pipeline_by_id(project_id, session)
            repo = PipelineRepository(session)
            updated = await repo.update(pipeline, partial_config)
            await session.commit()
            # notify source changes
            if pipeline.source != updated.source:
                self._notify_source_changed()
            if pipeline.status.is_running and updated.status.is_running:
                if pipeline.sink.id != updated.sink.id:  # type: ignore[union-attr] # sink is always there for running pipeline
                    await self._notify_sink_changed()
                # If the active model changes while running, notify inference to reload
                if pipeline.model.id != updated.model.id:  # type: ignore[union-attr]
                    self._model_service.activate_model()
            elif pipeline.status != updated.status:
                # If the pipeline is being activated or stopped
                await self._notify_pipeline_changed()
                # Intentionally call activate_model on status change regardless of whether a model exists.
                self._model_service.activate_model()
            if updated.inference_device != pipeline.inference_device:
                # reload model on device change
                self._model_service.activate_model()
            return updated

    @staticmethod
    async def get_active_pipeline() -> Pipeline | None:
        """Retrieve the currently active (running) pipeline from the database."""
        async with get_async_db_session_ctx() as session:
            return await PipelineRepository(session).get_active_pipeline()
