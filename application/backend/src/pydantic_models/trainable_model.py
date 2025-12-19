# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from pydantic import BaseModel, Field


class ModelFamily(StrEnum):
    """Model family/architecture type."""

    MEMORY_BANK = "memory_bank"
    DISTRIBUTION = "distribution"
    RECONSTRUCTION = "reconstruction"
    STUDENT_TEACHER = "student_teacher"
    GAN_BASED = "gan_based"
    TRANSFORMER = "transformer"
    FOUNDATION = "foundation"
    OTHER = "other"


class PerformanceMetrics(BaseModel):
    """Relative performance scores (1=low, 2=medium, 3=high). Null if not benchmarked."""

    training: int | None = Field(default=None, ge=1, le=3, description="Training speed score (3=fastest)")
    inference: int | None = Field(default=None, ge=1, le=3, description="Inference speed score (3=fastest)")


class TrainableModel(BaseModel):
    """Metadata for a trainable model template."""

    id: str = Field(description="Model identifier used in training API (snake_case)")
    name: str = Field(description="Human-readable model name")
    description: str | None = Field(default=None, description="Brief model description")
    family: ModelFamily | list[ModelFamily] = Field(default=ModelFamily.OTHER, description="Model architecture family")
    recommended: bool = Field(default=False, description="Whether model is recommended for new users")
    license: str = Field(default="Apache-2.0")
    docs_url: str | None = Field(default=None, description="Link to model documentation")
    metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    parameters: float | None = Field(default=None, description="Model parameters in millions")


class TrainableModelList(BaseModel):
    """List of trainable models with metadata."""

    trainable_models: list[TrainableModel]
