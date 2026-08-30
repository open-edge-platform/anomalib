# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from db import get_async_db_session_ctx
from pydantic_models import StartupProjectSelection, StartupProjectSelectionSource
from repositories import AppStateRepository, PipelineRepository, ProjectRepository
from services.exceptions import ResourceNotFoundError, ResourceType
from utils.short_uuid import ShortUUID


async def _resolve_last_used(
    app_state_repo: AppStateRepository,
    project_repo: ProjectRepository,
) -> StartupProjectSelection | None:
    last_used_project_id = await app_state_repo.get_last_used_project_id()
    if last_used_project_id is None:
        return None

    last_used_project = await project_repo.get_by_id(last_used_project_id)
    if last_used_project is not None:
        return StartupProjectSelection(
            project_id=last_used_project.id,
            source=StartupProjectSelectionSource.LAST_USED,
        )

    await app_state_repo.clear_last_used_project_id()
    return None


async def _resolve_active_pipeline(pipeline_repo: PipelineRepository) -> StartupProjectSelection | None:
    active_pipeline = await pipeline_repo.get_active_pipeline()
    if active_pipeline is None:
        return None
    return StartupProjectSelection(
        project_id=active_pipeline.project_id,
        source=StartupProjectSelectionSource.ACTIVE_PIPELINE,
    )


async def _resolve_first_project(project_repo: ProjectRepository) -> StartupProjectSelection | None:
    first_project = await project_repo.get_first_project()
    if first_project is None:
        return None
    return StartupProjectSelection(
        project_id=first_project.id,
        source=StartupProjectSelectionSource.FIRST_PROJECT,
    )


class ProjectSelectionService:
    @staticmethod
    async def get_startup_project_selection() -> StartupProjectSelection:
        """Resolve startup project in a deterministic order.

        Priority:
        1. last used project
        2. project that owns the active pipeline
        3. first project in the existing project list
        """
        async with get_async_db_session_ctx() as session:
            app_state_repo = AppStateRepository(session)
            project_repo = ProjectRepository(session)
            pipeline_repo = PipelineRepository(session)

            selection = await _resolve_last_used(app_state_repo, project_repo)

            if selection is None:
                selection = await _resolve_active_pipeline(pipeline_repo)

            if selection is None:
                selection = await _resolve_first_project(project_repo)

            if selection is None:
                selection = StartupProjectSelection(source=StartupProjectSelectionSource.NONE)

            return selection

    @staticmethod
    async def set_last_used_project(project_id: ShortUUID) -> None:
        async with get_async_db_session_ctx() as session:
            project_repo = ProjectRepository(session)
            if await project_repo.get_by_id(project_id) is None:
                raise ResourceNotFoundError(resource_type=ResourceType.PROJECT, resource_id=str(project_id))

            app_state_repo = AppStateRepository(session)
            await app_state_repo.set_last_used_project_id(str(project_id))
