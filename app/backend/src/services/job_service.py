# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from db.engine import get_async_db_session_ctx
from exceptions import DuplicateJobException, ResourceNotFoundException
from pydantic_models import Job, JobList, JobType
from pydantic_models.job import JobStatus, JobSubmitted, TrainJobPayload
from repositories import JobRepository


class JobService:
    @staticmethod
    async def get_job_list() -> JobList:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return JobList(jobs=await repo.get_all())

    async def get_job_by_id(self, job_id: UUID) -> Job | None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return await repo.get_by_id(job_id)

    async def submit_train_job(self, payload: TrainJobPayload) -> JobSubmitted:
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

    async def get_pending_train_job(self) -> Job | None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            return await repo.get_pending_job_by_type(JobType.TRAINING)

    async def update_job_status(self, job_id: UUID, status: JobStatus, message: str | None = None) -> None:
        async with get_async_db_session_ctx() as session:
            repo = JobRepository(session)
            job = await repo.get_by_id(job_id)
            if job is None:
                raise ResourceNotFoundException(resource_id=job_id, resource_name="job")
            job.status = status
            if message:
                job.message = message
            await repo.update(job)
