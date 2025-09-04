# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable

from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import ModelDB
from models import Model
from repositories.base import ProjectBaseRepository
from repositories.mappers import ModelMapper


class ModelRepository(ProjectBaseRepository):
    def __init__(self, db: AsyncSession, project_id: str):
        super().__init__(db, schema=ModelDB, project_id=project_id)

    @property
    def to_schema(self) -> Callable[[Model], ModelDB]:
        return ModelMapper.to_schema

    @property
    def from_schema(self) -> Callable[[ModelDB], Model]:
        return ModelMapper.from_schema
