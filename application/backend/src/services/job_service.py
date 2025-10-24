# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import datetime
import os
from uuid import UUID

import anyio
from sqlalchemy.exc import IntegrityError

from db import get_async_db_session_ctx
from exceptions import DuplicateJobException, ResourceNotFoundException
from pydantic_models import Job, JobList, JobType
from pydantic_models.job import JobStatus, JobSubmitted, TrainJobPayload
from repositories import JobRepository


class JobService:
    @staticmethod
    async def get_job_list(extra_filters: dict | None = None) -> JobList:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return JobList(jobs=await repo.get_all(extra_filters=extra_filters))

    @staticmethod
    async def get_job_by_id(job_id: UUID) -> Job | None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return await repo.get_by_id(job_id)

    @staticmethod
    async def submit_train_job(payload: TrainJobPayload) -> JobSubmitted:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            if await repo.is_job_duplicate(project_id=payload.project_id, payload=payload):
                raise DuplicateJobException

            try:
                job = Job(
                    project_id=payload.project_id,
                    type=JobType.TRAINING,
                    payload=payload.model_dump(),
                    message="Training job submitted",
                )
                saved_job = await repo.save(job)
                return JobSubmitted(job_id=saved_job.id)
            except IntegrityError:
                raise ResourceNotFoundException(resource_id=payload.project_id, resource_name="project")

    @staticmethod
    async def get_pending_train_job() -> Job | None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return await repo.get_pending_job_by_type(JobType.TRAINING)

    @staticmethod
    async def update_job_status(
        job_id: UUID, status: JobStatus, message: str | None = None, progress: int | None = None
    ) -> None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id(job_id)
            if job is None:
                raise ResourceNotFoundException(resource_id=job_id, resource_name="job")
            updates: dict = {"status": status}
            if message is not None:
                updates["message"] = message
            progress_ = 100 if status is JobStatus.COMPLETED else progress

            if status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED}:
                updates["end_time"] = datetime.datetime.now(tz=datetime.timezone.utc)

            if progress_ is not None:
                updates["progress"] = progress_
            await repo.update(job, updates)

    @classmethod
    async def stream_logs(cls, job_id: UUID | str):
        from core.logging.utils import get_job_logs_path

        log_file = get_job_logs_path(job_id=job_id)
        if not os.path.exists(log_file):
            raise ResourceNotFoundException(resource_id=job_id, resource_name="job_logs")

        async def is_job_still_running():
            job = await cls.get_job_by_id(job_id=job_id)
            if job is None:
                raise ResourceNotFoundException(resource_id=job_id, resource_name="job")
            return job.status == JobStatus.RUNNING

        # Cache job status and only check every 2 seconds
        status_check_interval = 2.0  # seconds
        last_status_check = 0.0
        cached_still_running = True
        loop = asyncio.get_running_loop()

        async with await anyio.open_file(log_file) as f:
            while True:
                line = await f.readline()
                now = loop.time()
                # Only check job status every status_check_interval seconds
                if now - last_status_check > status_check_interval:
                    cached_still_running = await is_job_still_running()
                    last_status_check = now
                still_running = cached_still_running
                if not line:
                    # wait for more lines if job is still running
                    if still_running:
                        await asyncio.sleep(0.5)
                        continue
                    # No more lines are expected
                    else:
                        yield "data: DONE\n\n"
                        break

                # Format as an SSE message
                yield f"data: {line.rstrip()}\n\n"
