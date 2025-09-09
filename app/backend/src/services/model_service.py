# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
from uuid import UUID

from sqlalchemy.ext.asyncio.session import AsyncSession

from anomalib.deploy import ExportType, OpenVINOInferencer
from models import Model, ModelList
from repositories import ModelRepository

from repositories.binary_repo import ModelBinaryRepository



class ModelService:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session

    def repository(self, project_id: str | UUID) -> ModelRepository:
        return ModelRepository(self.db_session, project_id=str(project_id))

    async def create_model(self, model: Model) -> Model:
        return await self.repository(model.project_id).save(model)

    async def get_model_list(self, project_id: UUID) -> ModelList:
        return ModelList(models=await self.repository(project_id).get_all())

    async def get_model_by_id(self, project_id: UUID, model_id: UUID) -> Model | None:
        return await self.repository(project_id).get_by_id(model_id)

    async def delete_model(self, project_id: UUID, model_id: UUID) -> None:
        return await self.repository(project_id).delete(model_id)

    async def load_inference_model(self, model: Model, device: str = "CPU") -> OpenVINOInferencer:
        if model.format is not ExportType.OPENVINO:
            raise NotImplementedError(f"Model format {model.format} is not supported for inference at this moment.")

        model_bin_repo = ModelBinaryRepository(project_id=model.project_id, model_id=model.id)
        model_path = model_bin_repo.get_weights_file_path(format=model.format, name="model.xml")
        return await asyncio.to_thread(OpenVINOInferencer, path=model_path, device="CPU")
