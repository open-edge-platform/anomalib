# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable

from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import SourceDB
from pydantic_models import Source
from repositories.base import BaseRepository
from repositories.mappers import SourceMapper


class SourceRepository(BaseRepository):
    """Repository for source-related database operations."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, SourceDB)

    @property
    def to_schema(self) -> Callable[[Source], SourceDB]:
        return SourceMapper.to_schema

    @property
    def from_schema(self) -> Callable[[SourceDB], Source]:
        return SourceMapper.from_schema
