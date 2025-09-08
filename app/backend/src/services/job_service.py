# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio.session import AsyncSession

from exceptions import DuplicateJobException, ResourceNotFoundException
from models import Job, JobList, JobType
from models.job import JobSubmitted, TrainJobPayload
from repositories import JobRepository


class JobService:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.job_repository = JobRepository(db_session)

    async def get_job_list(self) -> JobList:
        return JobList(jobs=await self.job_repository.get_all())

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
