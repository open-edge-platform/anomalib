# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse

from api.dependencies import get_job_id, get_job_service
from api.endpoints import API_PREFIX
from pydantic_models import JobList
from pydantic_models.job import JobCancelled, JobSubmitted, TrainJobPayload
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


@job_router.get("/{job_id}/progress")
async def get_job_progress(
    job_id: Annotated[UUID, Depends(get_job_id)],
    job_service: Annotated[JobService, Depends(get_job_service)],
) -> StreamingResponse:
    """Endpoint to get the progress of a job by its ID"""
    return StreamingResponse(job_service.stream_progress(job_id=job_id), media_type="text/event-stream")


@job_router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: Annotated[UUID, Depends(get_job_id)],
    job_service: Annotated[JobService, Depends(get_job_service)],
) -> JobCancelled:
    """Endpoint to cancel a job by its ID"""
    return await job_service.cancel_job(job_id=job_id)
