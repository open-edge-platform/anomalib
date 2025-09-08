# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import abc
from collections.abc import Callable
from typing import Any, TypeVar
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.sql import expression
from sqlalchemy.sql.selectable import Select, and_

from db.schema import Base

ModelType = TypeVar("ModelType", bound=BaseModel)
SchemaType = TypeVar("SchemaType", bound=Base)


class BaseRepository[ModelType, SchemaType](metaclass=abc.ABCMeta):
    """Base repository class for database operations."""

    def __init__(self, db: AsyncSession, schema: type[SchemaType]):
        self.db = db
        self.schema = schema

    @property
    @abc.abstractmethod
    def to_schema(self) -> Callable[[ModelType], SchemaType]:
        """to_schema mapper callable"""

    @property
    @abc.abstractmethod
    def from_schema(self) -> Callable[[SchemaType], ModelType]:
        """from_schema mapper callable"""

    @property
    def base_filters(self) -> dict:
        """Base filter expression for the repository"""
        return {}

    def _get_filter_query(self, extra_filters: dict | None = None, expressions: list[Any] | None = None) -> Select:
        """Build query with filters and expressions combined with AND."""
        query = expression.select(self.schema)

        # Apply keyword filters (column=value)
        if extra_filters is None:
            extra_filters = {}
        combined_filters = extra_filters | self.base_filters
        if combined_filters:
            query = query.filter_by(**combined_filters)

        # Apply additional expressions with AND
        if expressions:
            query = query.where(and_(*expressions))

        return query

    async def get_by_id(self, obj_id: str | UUID) -> ModelType | None:
        return await self.get_one(extra_filters={"id": self._id_to_str(obj_id)})

    async def get_one(
        self, extra_filters: dict | None = None, expressions: list[Any] | None = None
    ) -> ModelType | None:
        query = self._get_filter_query(extra_filters=extra_filters, expressions=expressions)
        result = await self.db.execute(query)
        first_result = result.scalars().first()
        if first_result:
            return self.from_schema(first_result)
        return None

    async def get_all(self, extra_filters: dict | None = None, expressions: list[Any] | None = None) -> list[ModelType]:
        query = self._get_filter_query(extra_filters=extra_filters, expressions=expressions)
        results = await self.db.execute(query)
        scalars = results.scalars().all()
        return [self.from_schema(result) for result in scalars]

    async def save(self, item: ModelType) -> ModelType:
        schema_item: SchemaType = self.to_schema(item)
        self.db.add(schema_item)
        await self.db.commit()
        return item

    async def update(self, item: ModelType) -> ModelType:
        schema_item: SchemaType = self.to_schema(item)
        await self.db.merge(schema_item)
        await self.db.commit()
        return item

    async def delete(self, obj_id: str | UUID) -> None:
        obj_id = self._id_to_str(obj_id)
        query = expression.delete(self.schema).where(self.schema.id == obj_id, self.base_filter_expression)  # type: ignore[attr-defined]
        await self.db.execute(query)

    @staticmethod
    def _id_to_str(obj_id: str | UUID) -> str:
        if isinstance(obj_id, UUID):
            return str(obj_id)
        return obj_id


class ProjectBaseRepository(BaseRepository, metaclass=abc.ABCMeta):
    def __init__(self, db: AsyncSession, project_id: str | UUID, schema: type[SchemaType]):
        super().__init__(db, schema)
        self.project_id = self._id_to_str(project_id)

    @property
    def base_filters(self) -> dict:
        return {"project_id": self.project_id}
