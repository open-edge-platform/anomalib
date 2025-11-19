# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Endpoints for managing active pipelines"""

from typing import Annotated

from fastapi import APIRouter, Depends, status

from geti_inspect.api.dependencies import get_pipeline_service
from geti_inspect.pydantic_models.pipeline import Pipeline
from geti_inspect.services import PipelineService

router = APIRouter(prefix="/api/active-pipeline", tags=["Pipelines"])


@router.get(
    "",
    responses={
        status.HTTP_200_OK: {"description": "Active pipeline found"},
        status.HTTP_204_NO_CONTENT: {"description": "No active pipeline found"},
    },
    response_model_exclude_none=True,
)
async def get_active_pipeline(
    pipeline_service: Annotated[PipelineService, Depends(get_pipeline_service)],
) -> Pipeline | None:
    """Get info about the active pipeline"""
    return await pipeline_service.get_active_pipeline()
