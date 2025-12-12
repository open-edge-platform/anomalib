# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from pydantic import BaseModel


class ModelFamily(StrEnum):
    PATCH_BASED = "patch_based"
    MEMORY_BANK = "memory_bank"
    STUDENT_TEACHER = "student_teacher"
    RECONSTRUCTION_BASED = "reconstruction_based"
    DISTRIBUTION_MAP = "distribution_map"


class TrainingTime(StrEnum):
    COFFEE = "coffee"
    WALK = "walk"
    CYCLE = "cycle"


class ModelInfo(BaseModel):
    name: str
    class_name: str
    training_time: TrainingTime
    model_family: list[ModelFamily]
    recommended: bool = False
    license: str = "Apache-2.0"


class TrainableModelList(BaseModel):
    """List wrapper for returning multiple trainable models."""

    trainable_models: list[ModelInfo]
