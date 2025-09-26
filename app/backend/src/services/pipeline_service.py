# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import statistics
from datetime import UTC, datetime, timedelta
from multiprocessing.synchronize import Condition
from uuid import UUID

from sqlalchemy.ext.asyncio.session import AsyncSession

from db import get_async_db_session_ctx
from exceptions import ResourceNotFoundException
from pydantic_models import Pipeline, PipelineStatus
from pydantic_models.metrics import InferenceMetrics, LatencyMetrics, PipelineMetrics, TimeWindow
from repositories import PipelineRepository
from services import ActivePipelineService
from services.metrics_service import MetricsService
from services.model_service import ModelService

MSG_ERR_DELETE_RUNNING_PIPELINE = "Cannot delete a running pipeline."


class PipelineService:
    def __init__(
        self,
        active_pipeline_service: ActivePipelineService,
        metrics_service: MetricsService,
        config_changed_condition: Condition,
        model_service: ModelService,
    ) -> None:
        self._active_pipeline_service: ActivePipelineService = active_pipeline_service
        self._config_changed_condition: Condition = config_changed_condition
        self._metrics_service: MetricsService = metrics_service
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
            if pipeline.status == PipelineStatus.RUNNING and updated.status == PipelineStatus.RUNNING:
                # If the pipeline source_id or sink_id is being updated while running
                if pipeline.source.id != updated.source.id:  # type: ignore[union-attr] # source is always there for running pipeline
                    self._notify_source_changed()
                if pipeline.sink.id != updated.sink.id:  # type: ignore[union-attr] # sink is always there for running pipeline
                    await self._notify_sink_changed()
                # If the active model changes while running, notify inference to reload
                if pipeline.model.id != updated.model.id:  # type: ignore[union-attr]
                    self._model_service.activate_model()
            elif pipeline.status != updated.status:
                # If the pipeline is being activated or stopped
                await self._notify_pipeline_changed()
                # On activation, trigger model activation so inference reloads the active model
                if updated.status == PipelineStatus.RUNNING and updated.model is not None:
                    self._model_service.activate_model()
            return updated

    async def get_pipeline_metrics(self, pipeline_id: UUID, time_window: int = 60) -> PipelineMetrics:
        """Calculate metrics for a pipeline over a specified time window."""
        # First check if pipeline exists
        pipeline = await self.get_pipeline_by_id(pipeline_id)
        if pipeline.status != PipelineStatus.RUNNING:
            raise ValueError("Cannot get metrics for a pipeline that is not running.")

        # Calculate time window
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(seconds=time_window)

        # Get actual latency measurements from the metrics service
        latency_samples = self._metrics_service.get_latency_measurements(
            model_id=pipeline.model_id,  # type: ignore[arg-type] # model is always there for running pipeline
            time_window=time_window,
        )

        # Calculate latency metrics
        if latency_samples:
            latency_metrics = LatencyMetrics(
                avg_ms=statistics.mean(latency_samples),
                min_ms=min(latency_samples),
                max_ms=max(latency_samples),
                p95_ms=self._calculate_percentile(latency_samples, 95),
                latest_ms=latency_samples[-1],
            )
        else:
            # No data available
            latency_metrics = LatencyMetrics(avg_ms=None, min_ms=None, max_ms=None, p95_ms=None, latest_ms=None)

        window = TimeWindow(start=start_time, end=end_time, time_window=time_window)
        inference_metrics = InferenceMetrics(latency=latency_metrics)
        return PipelineMetrics(time_window=window, inference=inference_metrics)

    @staticmethod
    def _calculate_percentile(data: list[float], percentile: int) -> float:
        """Calculate the specified percentile of the data."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (percentile / 100.0)
        floor_k = int(k)
        ceil_k = floor_k + 1

        if ceil_k >= len(sorted_data):
            return sorted_data[-1]

        # Linear interpolation
        fraction = k - floor_k
        return sorted_data[floor_k] + fraction * (sorted_data[ceil_k] - sorted_data[floor_k])
