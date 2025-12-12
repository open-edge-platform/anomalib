# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache

from anomalib.models import list_models
from fastapi import APIRouter

from api.endpoints import API_PREFIX
from pydantic_models import TrainableModelList

router = APIRouter(prefix=f"{API_PREFIX}/trainable-models", tags=["Trainable Models"])


@lru_cache
def _get_trainable_models() -> TrainableModelList:  # pragma: no cover
    """Return list of trainable models with optional descriptions.

    The available models are retrieved from ``anomalib.models.list_models``. Currently, only
    the model names are returned. Descriptions can be added manually in the
    ``_MODEL_DESCRIPTIONS`` mapping below.
    """
    model_names = sorted(list_models(case="pascal"))

    return TrainableModelList(trainable_models=model_names)


@router.get("", summary="List trainable models")
async def list_trainable_models() -> TrainableModelList:
    """GET endpoint returning available trainable model names."""

    return _get_trainable_models()
