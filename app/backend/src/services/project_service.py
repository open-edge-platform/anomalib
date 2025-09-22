# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from db import get_async_db_session_ctx
from pydantic_models import Project, ProjectList
from repositories import ProjectRepository


class ProjectService:
    @staticmethod
    async def get_project_list() -> ProjectList:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            return ProjectList(projects=await repo.get_all())

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
    async def delete_project(project_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            repo = ProjectRepository(session)
            await repo.delete_by_id(project_id)
