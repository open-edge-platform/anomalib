# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from sqlalchemy.ext.asyncio.session import AsyncSession

from models import Project, ProjectList
from repositories import ProjectRepository


class ProjectService:
    def __init__(self, db_session: AsyncSession):
        self.project_repository = ProjectRepository(db_session)

    async def get_project_list(self) -> ProjectList:
        return ProjectList(projects=await self.project_repository.get_all())

    async def get_project_by_id(self, project_id: UUID) -> Project | None:
        return await self.project_repository.get_by_id(project_id)

    async def create_project(self, project: Project) -> Project:
        return await self.project_repository.save(project)

    async def delete_project(self, project_id: UUID) -> None:
        await self.project_repository.delete(project_id)
