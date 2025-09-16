# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio.session import AsyncSession

from exceptions import DuplicateJobException, ResourceNotFoundException
from models import Job, JobList, JobType
from models.job import JobStatus, JobSubmitted, TrainJobPayload
from repositories import JobRepository


class JobService:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.job_repository = JobRepository(db_session)

    async def get_job_list(self) -> JobList:
        return JobList(jobs=await self.job_repository.get_all())

    async def get_job_by_id(self, job_id: UUID) -> Job | None:
        return await self.job_repository.get_by_id(job_id)

    async def submit_train_job(self, payload: TrainJobPayload) -> JobSubmitted:
        if await self.job_repository.is_job_duplicate(project_id=payload.project_id, payload=payload):
            raise DuplicateJobException

        try:
            job = Job(
                project_id=payload.project_id,
                type=JobType.TRAINING,
                payload=payload.model_dump(),
                message="Training job submitted",
            )
            saved_job = await self.job_repository.save(job)
            return JobSubmitted(job_id=saved_job.id)
        except IntegrityError:
            raise ResourceNotFoundException(resource_id=payload.project_id, resource_name="project")

    async def get_pending_train_job(self) -> Job | None:
        return await self.job_repository.get_pending_job_by_type(JobType.TRAINING)

    async def update_job_status(self, job_id: UUID, status: JobStatus, message: str | None = None) -> None:
        job = await self.job_repository.get_by_id(job_id)
        if job is None:
            raise ResourceNotFoundException(resource_id=job_id, resource_name="job")
        job.status = status
        if message:
            job.message = message
        await self.job_repository.update(job)
