# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from sqlalchemy.ext.asyncio.session import AsyncSession

from models import Model, ModelList
from repositories import ModelRepository


class ModelService:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    def repository(self, project_id: str | UUID) -> ModelRepository:
        return ModelRepository(self.db_session, project_id=str(project_id))

    async def get_model_list(self, project_id: UUID) -> ModelList:
        return ModelList(models=await self.repository(project_id).get_all())

    async def get_model_by_id(self, project_id: UUID, model_id: UUID) -> Model | None:
        return await self.repository(project_id).get_by_id(model_id)

    async def delete_model(self, project_id: UUID, model_id: UUID) -> None:
        return await self.repository(project_id).delete(model_id)
