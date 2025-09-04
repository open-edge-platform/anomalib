# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from models.base import BaseIDNameModel


class ModelFormat(StrEnum):
    OPENVINO = "openvino_ir"
    ONNX = "onnx"


class Model(BaseIDNameModel):
    """
    Base model schema that includes common fields for all models.
    This can be extended by other schemas to include additional fields.
    """

    format: ModelFormat = ModelFormat.OPENVINO
    project_id: UUID
    threshold: float = Field(default=0.5, gt=0.0, lt=1.0, description="Confidence threshold for the model")
    is_ready: bool = Field(default=False, description="Indicates if the model is ready for use")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "76e07d18-196e-4e33-bf98-ac1d35dca4cb",
                "name": "YOLO-X for Vehicle Detection",
                "format": "openvino_ir",
            }
        }
    }


class ModelList(BaseModel):
    models: list[Model]
