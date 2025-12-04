# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from db import get_async_db_session_ctx
from pydantic_models import Project, ProjectList, ProjectUpdate
from pydantic_models.base import Pagination
from repositories import ProjectRepository


class ProjectService:
    @staticmethod
    async def get_project_list(limit: int, offset: int) -> ProjectList:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            total = await repo.get_all_count()
            items = await repo.get_all_pagination(limit=limit, offset=offset)
        return ProjectList(
            projects=items,
            pagination=Pagination(
                limit=limit,
                offset=offset,
                count=len(items),
                total=total,
            ),
        )

    @staticmethod
    async def get_project_by_id(project_id: UUID) -> Project | None:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            return await repo.get_by_id(project_id)

    @staticmethod
    async def create_project(project: Project) -> Project:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            return await repo.save(project)

    @staticmethod
    async def update_project(project_id: UUID, project_update: ProjectUpdate) -> Project | None:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            project = await repo.get_by_id(project_id)
            if project is None:
                return None
            return await repo.update(project, project_update.model_dump(exclude_unset=True))

    @staticmethod
    async def delete_project(project_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            await repo.delete_by_id(project_id)
