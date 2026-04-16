# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from pathlib import Path

import yaml
from anomalib.models import list_models
from fastapi import APIRouter
from loguru import logger

from api.endpoints import API_PREFIX
from pydantic_models import TrainableModel, TrainableModelList

router = APIRouter(prefix=f"{API_PREFIX}/trainable-models", tags=["Trainable Models"])

CORE_DIR = Path(__file__).parent.parent.parent / "core"


@lru_cache
def _load_model_metadata() -> TrainableModelList:
    """Load and cache model metadata from YAML configuration file.

    The model metadata is stored in ``core/model_metadata.yaml`` and contains
    information about each model including display name, description, family,
    performance metrics, and whether it is recommended/supported.

    Models are validated against anomalib's available models list. Any model
    in the metadata that doesn't exist in anomalib will be logged as a warning
    and excluded from the results.

    Returns:
        TrainableModelList containing only supported models that exist in
        anomalib, sorted by recommended first, then by name.
    """
    metadata_path = CORE_DIR / "model_metadata.yaml"

    with open(metadata_path, encoding="utf-8") as f:
        raw_models = yaml.safe_load(f)

    if not raw_models:
        return TrainableModelList(trainable_models=[])

    available_models = set(list_models(case="snake"))

    models = []
    for m in raw_models:
        if not m.get("supported", True):
            continue

        model_id = m.get("id", "")
        if model_id not in available_models:
            logger.warning(f"Model '{model_id}' from metadata not found in anomalib.models.list_models()")
            continue

        models.append(TrainableModel(**m))

    models.sort(key=lambda m: (not m.recommended, m.name))

    return TrainableModelList(trainable_models=models)


@router.get("", summary="List trainable models")
async def list_trainable_models() -> TrainableModelList:
    """GET endpoint returning available trainable model metadata.

    Returns a list of trainable models with their metadata including:
    - id: Model identifier for training API
    - name: Human-readable display name
    - description: Brief model description
    - family: Model architecture family (memory_bank, distribution, etc.)
    - recommended: Whether model is recommended for new users
    - metrics: Training and inference speed scores (1-3 scale)
    - parameters: Model size in millions of parameters
    """
    return _load_model_metadata()
