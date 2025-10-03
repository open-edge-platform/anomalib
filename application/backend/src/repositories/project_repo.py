# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable

from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import PipelineDB, ProjectDB
from pydantic_models import Project
from repositories.base import BaseRepository
from repositories.mappers import ProjectMapper


class ProjectRepository(BaseRepository):
    def __init__(self, db: AsyncSession):
        super().__init__(db, ProjectDB)

    @property
    def to_schema(self) -> Callable[[Project], ProjectDB]:
        return ProjectMapper.to_schema

    @property
    def from_schema(self) -> Callable[[ProjectDB], Project]:
        return ProjectMapper.from_schema

    async def get_by_name(self, name: str) -> list[Project]:
        return await self.get_all(extra_filters={"name": name})

    async def save(self, project: Project) -> Project:
        project_schema: ProjectDB = self.to_schema(project)
        project_schema.pipeline = PipelineDB(
            project_id=project_schema.id,
        )
        self.db.add(project_schema)
        await self.db.commit()
        return project
