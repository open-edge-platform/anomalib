# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, computed_field, field_serializer

from pydantic_models.base import BaseIDModel


class JobType(StrEnum):
    TRAINING = "training"
    OPTIMIZATION = "optimization"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class JobStage(StrEnum):
    """Job stages follow PyTorch Lightning stages with the addition of idle stage.

    See ``lightning.pytorch.trainer.states.RunningStage`` for more details.
    """

    IDLE = "idle"
    TRAINING = "train"
    SANITY_CHECKING = "sanity_check"
    VALIDATING = "validate"
    TESTING = "test"
    PREDICTING = "predict"


class Job(BaseIDModel):
    project_id: UUID
    type: JobType = JobType.TRAINING
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage from 0 to 100")
    status: JobStatus = JobStatus.PENDING
    stage: JobStage = JobStage.IDLE
    payload: dict
    message: str = "Job created"
    start_time: datetime | None = None
    end_time: datetime | None = None

    @field_serializer("project_id")
    def serialize_project_id(self, project_id: UUID, _info: Any) -> str:
        return str(project_id)


class JobList(BaseModel):
    jobs: list[Job]


class JobSubmitted(BaseModel):
    job_id: UUID


class JobCancelled(BaseModel):
    job_id: UUID

    @computed_field
    def message(self) -> str:
        return f"Job with ID `{self.job_id}' marked as cancelled."


class TrainJobPayload(BaseModel):
    project_id: UUID = Field(exclude=True)
    model_name: str
    device: str | None = None
