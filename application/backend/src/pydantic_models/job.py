# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_serializer

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


class Job(BaseIDModel):
    project_id: UUID
    type: JobType = JobType.TRAINING
    progress: int = Field(default=0, ge=0, le=100, description="Progress percentage from 0 to 100")
    status: JobStatus = JobStatus.PENDING
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


class TrainJobPayload(BaseModel):
    project_id: UUID = Field(exclude=True)
    model_name: str
    device: str | None
