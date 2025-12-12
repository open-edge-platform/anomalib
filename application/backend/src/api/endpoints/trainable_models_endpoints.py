# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from fastapi import APIRouter

from api.endpoints import API_PREFIX
from pydantic_models import ModelFamily, ModelInfo, TrainableModelList, TrainingTime

router = APIRouter(prefix=f"{API_PREFIX}/trainable-models", tags=["Trainable Models"])


def _get_trainable_models() -> TrainableModelList:  # pragma: no cover
    """Return list of trainable models with optional descriptions.

    Currently hardcoded for v1.
    """
    models = [
        ModelInfo(
            name="PatchCore",
            class_name="patchcore",
            training_time=TrainingTime.COFFEE,
            model_family=[ModelFamily.MEMORY_BANK, ModelFamily.PATCH_BASED],
            recommended=True,
        ),
        ModelInfo(
            name="FRE",
            class_name="fre",
            training_time=TrainingTime.COFFEE,
            model_family=[ModelFamily.RECONSTRUCTION_BASED],
            recommended=True,
        ),
        ModelInfo(
            name="Dinomaly",
            class_name="dinomaly",
            training_time=TrainingTime.CYCLE,
            model_family=[ModelFamily.STUDENT_TEACHER, ModelFamily.RECONSTRUCTION_BASED],
            recommended=True,
        ),
        ModelInfo(
            name="CFA",
            class_name="cfa",
            training_time=TrainingTime.COFFEE,
            model_family=[ModelFamily.MEMORY_BANK],
        ),
        ModelInfo(
            name="DFM",
            class_name="dfm",
            training_time=TrainingTime.COFFEE,
            model_family=[ModelFamily.MEMORY_BANK],
        ),
        ModelInfo(
            name="FastFlow",
            class_name="fastflow",
            training_time=TrainingTime.CYCLE,
            model_family=[ModelFamily.DISTRIBUTION_MAP],
        ),
        ModelInfo(
            name="Padim",
            class_name="padim",
            training_time=TrainingTime.COFFEE,
            model_family=[ModelFamily.MEMORY_BANK, ModelFamily.PATCH_BASED, ModelFamily.DISTRIBUTION_MAP],
        ),
        ModelInfo(
            name="Reverse Distillation",
            class_name="reverse_distillation",
            training_time=TrainingTime.CYCLE,
            model_family=[ModelFamily.STUDENT_TEACHER],
        ),
        ModelInfo(
            name="SuperSimpleNet",
            class_name="supersimplenet",
            training_time=TrainingTime.COFFEE,
            model_family=[ModelFamily.RECONSTRUCTION_BASED],
        ),
    ]

    return TrainableModelList(trainable_models=models)


@router.get("", summary="List trainable models")
async def list_trainable_models() -> TrainableModelList:
    """GET endpoint returning available trainable model names."""

    return _get_trainable_models()
