# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from abc import ABCMeta, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from db.schema import Base

DBEntity = TypeVar("DBEntity", bound=Base)
ModelEntity = TypeVar("ModelEntity", bound=BaseModel)


class IBaseMapper[DBEntity, ModelEntity](metaclass=ABCMeta):
    """Mapper for pydantic model entity <-> DB schema conversions."""

    @staticmethod
    @abstractmethod
    def to_schema(db_schema: ModelEntity) -> DBEntity:
        """Convert ModelEntity to DBEntity."""

    @staticmethod
    @abstractmethod
    def from_schema(model: DBEntity) -> ModelEntity:
        """Convert DBEntity to ModelEntity."""
