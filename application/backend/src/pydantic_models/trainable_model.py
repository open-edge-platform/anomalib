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


class License(StrEnum):
    """Model license type."""

    APACHE_2_0 = "Apache-2.0"
    MIT = "MIT"
    CC_BY_4_0 = "CC-BY-4.0"
    CC_BY_NC_4_0 = "CC-BY-NC-4.0"


class TrainableModel(BaseModel):
    """Metadata for a trainable model template."""

    id: str = Field(description="Model identifier used in training API (snake_case)")
    name: str = Field(description="Human-readable model name")
    description: str = Field(description="Brief model description")
    family: list[ModelFamily] = Field(default_factory=list, description="Model architecture family")
    recommended: bool = Field(default=False, description="Whether model is recommended for new users")
    license: License = Field(default=License.APACHE_2_0, description="Model license type")
    docs_url: str | None = Field(default=None, description="Link to model documentation")
    parameters: float | None = Field(default=None, description="Model parameters in millions")


class TrainableModelList(BaseModel):
    """List of trainable models with metadata."""

    trainable_models: list[TrainableModel]
