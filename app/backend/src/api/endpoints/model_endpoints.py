# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import base64
import io
import logging
import cv2
from typing import Annotated
from uuid import UUID

import numpy as np
from PIL import Image
from fastapi import APIRouter, Depends, Request, UploadFile, Response

from api.media_rest_validator import MediaRestValidator
from exceptions import ResourceNotFoundException
from models import Model, ModelList
from api.dependencies import get_model_id, get_model_service, get_project_id
from api.endpoints.project_endpoints import project_api_prefix_url
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


@model_router.post("/{model_id}:predict")
async def predict(
    request: Request,
    model_service: Annotated[ModelService, Depends(get_model_service)],
    project_id: Annotated[UUID, Depends(get_project_id)],
    model_id: Annotated[UUID, Depends(get_model_id)],
    file: UploadFile = Depends(MediaRestValidator.validate_image_file),
):
    """Endpoint to run prediction using the specified model"""
    inference_model = request.app.state.active_models.get(model_id)
    if inference_model is None:
        model = await model_service.get_model_by_id(project_id=project_id, model_id=model_id)
        if model is None:
            raise ResourceNotFoundException(resource_id=model_id, resource_name="model")
        inference_model = await model_service.load_inference_model(model=model)
        request.app.state.active_models[model_id] = inference_model

    image_bytes = await file.read()
    npd = np.frombuffer(image_bytes, np.uint8)
    bgr_image = cv2.imdecode(npd, -1)
    numpy_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    pred = inference_model.predict(numpy_image)
    arr = pred.anomaly_map.squeeze()  # Remove dimensions of size 1
    arr_normalized = (arr * 255).astype(np.uint8)  # Normalize to 0-255 and convert to uint8
    im = Image.fromarray(arr_normalized, mode='L')  # 'L' for grayscale

    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        im.save(buf, format='PNG')
        im_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Create response with all prediction data
    label_str = "Anomalous" if pred.pred_label.item() else "Normal"
    response_data = {
        "anomaly_map": im_base64,
        "label": label_str,
        "score": float(pred.pred_score.item())
    }
    return response_data
