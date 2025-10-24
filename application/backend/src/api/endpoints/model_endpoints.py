# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status

from api.dependencies import get_device_name, get_model_id, get_model_service, get_project_id
from api.endpoints.project_endpoints import project_api_prefix_url
from api.media_rest_validator import MediaRestValidator
from exceptions import ResourceNotFoundException
from pydantic_models import Model, ModelList, PredictionResponse
from services import ModelService
from services.exceptions import DeviceNotFoundError

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


@model_router.post("/{model_id}:predict")
async def predict(
    request: Request,
    model_service: Annotated[ModelService, Depends(get_model_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
    model_id: Annotated[UUID, Depends(get_model_id)],
    file: Annotated[UploadFile, Depends(MediaRestValidator.validate_image_file)],
    device: Annotated[str | None, Depends(get_device_name)] = None,
) -> PredictionResponse:
    """
    Run prediction on an uploaded image using the specified model.

    Returns prediction results including anomaly map, label, and confidence score.
    """
    # Get model from database
    model = await model_service.get_model_by_id(project_id=project_id, model_id=model_id)
    if model is None:
        raise ResourceNotFoundException(resource_id=model_id, resource_name="model")

    # Read uploaded image and run prediction with model caching
    # Models are cached in request.app.state.active_models for performance
    image_bytes = await file.read()
    try:
        return await model_service.predict_image(model, image_bytes, request.app.state.active_models, device=device)
    except DeviceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
