# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from uuid import UUID

from sqlalchemy.ext.asyncio.session import AsyncSession

from db.schema import DatasetSnapshotDB
from pydantic_models.dataset_snapshot import DatasetSnapshot
from repositories.base import ProjectBaseRepository
from repositories.mappers.dataset_snapshot_mapper import DatasetSnapshotMapper


class DatasetSnapshotRepository(ProjectBaseRepository[DatasetSnapshot, DatasetSnapshotDB]):
    """Repository for Dataset Snapshot operations."""

    def __init__(self, db: AsyncSession, project_id: str | UUID):
        super().__init__(db, schema=DatasetSnapshotDB, project_id=project_id)

    @property
    def to_schema(self) -> Callable[[DatasetSnapshot], DatasetSnapshotDB]:
        return DatasetSnapshotMapper.to_schema

    @property
    def from_schema(self) -> Callable[[DatasetSnapshotDB], DatasetSnapshot]:
        return DatasetSnapshotMapper.from_schema
