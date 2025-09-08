# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Annotated

from fastapi import APIRouter, Body, Depends

from models import JobList
from models.job import JobSubmitted, TrainJobPayload
from rest_api.dependencies import get_job_service
from rest_api.endpoints import API_PREFIX
from services import JobService

logger = logging.getLogger(__name__)

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
