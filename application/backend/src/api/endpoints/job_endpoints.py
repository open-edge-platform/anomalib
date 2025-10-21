# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse

from api.dependencies import get_job_id, get_job_service
from api.endpoints import API_PREFIX
from pydantic_models import JobList
from pydantic_models.job import JobSubmitted, TrainJobPayload
from services import JobService

job_api_prefix_url = API_PREFIX + "/jobs"
job_router = APIRouter(
    prefix=job_api_prefix_url,
    tags=["Job"],
)


@job_router.get("", response_model_exclude_none=True)
async def get_jobs(job_service: Annotated[JobService, Depends(get_job_service)]) -> JobList:
    """Endpoint to get list of all jobs"""
    return await job_service.get_job_list()


@job_router.post(":train")
async def submit_train_job(
    job_service: Annotated[JobService, Depends(get_job_service)],
    payload: Annotated[TrainJobPayload, Body()],
) -> JobSubmitted:
    """Endpoint to submit a training job"""
    return await job_service.submit_train_job(payload=payload)


@job_router.get("/{job_id}/logs")
async def get_job_logs(
    job_id: Annotated[UUID, Depends(get_job_id)],
    job_service: Annotated[JobService, Depends(get_job_service)],
) -> StreamingResponse:
    """Endpoint to get the logs of a job by its ID"""
    return StreamingResponse(job_service.stream_logs(job_id=job_id), media_type="text/event-stream")
