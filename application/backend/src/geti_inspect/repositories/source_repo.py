# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from uuid import UUID

from sqlalchemy.ext.asyncio.session import AsyncSession

from geti_inspect.db.schema import SourceDB
from geti_inspect.pydantic_models import Source
from geti_inspect.repositories.base import ProjectBaseRepository
from geti_inspect.repositories.mappers import SourceMapper


class SourceRepository(ProjectBaseRepository):
    """Repository for source-related database operations."""

    def __init__(self, db: AsyncSession, project_id: UUID):
        super().__init__(db, schema=SourceDB, project_id=project_id)

    @property
    def to_schema(self) -> Callable[[Source], SourceDB]:
        return SourceMapper.to_schema

    @property
    def from_schema(self) -> Callable[[SourceDB], Source]:
        return SourceMapper.from_schema
