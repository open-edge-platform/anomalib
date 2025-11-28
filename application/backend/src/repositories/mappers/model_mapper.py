# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from db.schema import ModelDB
from pydantic_models import Model
from repositories.mappers.base_mapper_interface import IBaseMapper


class ModelMapper(IBaseMapper):
    @staticmethod
    def to_schema(model: Model) -> ModelDB:
        return ModelDB(**model.model_dump(mode="json"))

    @staticmethod
    def from_schema(model_db: ModelDB) -> Model:
        return Model.model_validate(model_db, from_attributes=True)
