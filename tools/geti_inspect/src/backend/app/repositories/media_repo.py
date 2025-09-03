# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from typing import Callable

from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import MediaDB
from models import Media
from repositories.base import ProjectBaseRepository
from repositories.mappers import MediaMapper


class MediaRepository(ProjectBaseRepository[Media]):
    def __init__(self, db: AsyncSession, project_id: str):
        super().__init__(db, schema=MediaDB, project_id=project_id)

    @property
    def to_schema(self) -> Callable[[Media], MediaDB]:
        return MediaMapper.to_schema

    @property
    def from_schema(self) -> Callable[[MediaDB], Media]:
        return MediaMapper.from_schema
