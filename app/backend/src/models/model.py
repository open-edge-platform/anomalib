# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
from uuid import UUID

from pydantic import BaseModel, Field

from anomalib.deploy import ExportType
from models.base import BaseIDNameModel


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
        return os.path.join(self.export_path, self.format.name.lower())

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
