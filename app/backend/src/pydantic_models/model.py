# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
from enum import Enum
from uuid import UUID

from anomalib.deploy import ExportType
from pydantic import BaseModel, Field, model_validator

from pydantic_models.base import BaseIDNameModel


class PredictionLabel(str, Enum):
    """Enum for prediction labels."""

    NORMAL = "Normal"
    ANOMALOUS = "Anomalous"


class Model(BaseIDNameModel):
    """
    Base model schema that includes common fields for all models.
    This can be extended by other schemas to include additional fields.
    """

    format: ExportType = ExportType.OPENVINO
    project_id: UUID
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0, description="Confidence threshold for the model")
    is_ready: bool = Field(default=False, description="Indicates if the model is ready for use")
    export_path: str | None = None

    @property
    def weights_path(self) -> str:
        if self.export_path is None:
            raise ValueError("export_path is required to get weights_path")
        return os.path.join(self.export_path, self.format.name.lower())

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "76e07d18-196e-4e33-bf98-ac1d35dca4cb",
                "name": "PatchCore",
                "format": "openvino",
            }
        }
    }


class ModelList(BaseModel):
    models: list[Model]


class PredictionResponse(BaseModel):
    """Response model for model prediction results."""

    anomaly_map: str = Field(description="Base64-encoded anomaly map image")
    label: PredictionLabel = Field(description="Prediction label")
    score: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")

    @model_validator(mode="after")
    def validate_score_range(self) -> "PredictionResponse":
        """Ensure score is within valid range [0, 1] and handle edge cases."""
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "anomaly_map": (
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                ),
                "label": "Normal",
                "score": 0.23,
            }
        }
    }
