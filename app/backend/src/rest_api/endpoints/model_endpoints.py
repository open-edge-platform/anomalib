# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends

from exceptions import ResourceNotFoundException
from models import Model, ModelList
from rest_api.dependencies import get_model_id, get_model_service, get_project_id
from rest_api.endpoints.project_endpoints import project_api_prefix_url
from services import ModelService

logger = logging.getLogger(__name__)

model_api_prefix_url = project_api_prefix_url + "/{project_id}/models"
model_router = APIRouter(
    prefix=model_api_prefix_url,
    tags=["Model"],
)


@model_router.get("")
async def get_models(
    model_service: Annotated[ModelService, Depends(get_model_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
) -> ModelList:
    """Endpoint to get list of all models"""
    return await model_service.get_model_list(project_id=project_id)


@model_router.get("/{model_id}")
async def get_model_info_by_id(
    model_service: Annotated[ModelService, Depends(get_model_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
    model_id: Annotated[UUID, Depends(get_model_id)],
) -> Model:
    """Endpoint to get model metadata by ID"""
    model = await model_service.get_model_by_id(project_id=project_id, model_id=model_id)
    if model is None:
        raise ResourceNotFoundException(resource_id=model_id, resource_name="model")
    return model
