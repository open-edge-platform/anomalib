# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from exceptions import DuplicateJobException, ResourceNotFoundException
from models import JobStatus, JobType
from repositories import JobRepository
from services import JobService


@pytest.fixture
def fxt_job_repository():
    """Fixture for a mock job repository."""
    return MagicMock(spec=JobRepository)


@pytest.fixture
def fxt_job_service(fxt_db_session, fxt_job_repository):
    """Fixture for JobService with mocked repository."""
    service = JobService(fxt_db_session)
    service.job_repository = fxt_job_repository
    return service


class TestJobService:
    def test_get_job_list(self, fxt_job_service, fxt_job_repository, fxt_job_list):
        """Test getting job list."""
        fxt_job_repository.get_all.return_value = fxt_job_list.jobs

        result = asyncio.run(fxt_job_service.get_job_list())

        assert result == fxt_job_list
        fxt_job_repository.get_all.assert_called_once()

    def test_get_job_by_id(self, fxt_job_service, fxt_job_repository, fxt_job):
        """Test getting job by ID."""
        fxt_job_repository.get_by_id.return_value = fxt_job

        result = asyncio.run(fxt_job_service.get_job_by_id(fxt_job.id))

        assert result == fxt_job
        fxt_job_repository.get_by_id.assert_called_once_with(fxt_job.id)

    def test_get_job_by_id_not_found(self, fxt_job_service, fxt_job_repository):
        """Test getting job by ID when not found."""
        fxt_job_repository.get_by_id.return_value = None

        result = asyncio.run(fxt_job_service.get_job_by_id("non-existent-id"))

        assert result is None
        fxt_job_repository.get_by_id.assert_called_once_with("non-existent-id")

    def test_submit_train_job_success(self, fxt_job_service, fxt_job_repository, fxt_job_payload, fxt_job):
        """Test successful job submission."""
        fxt_job_repository.is_job_duplicate.return_value = False
        fxt_job_repository.save.return_value = fxt_job

        result = asyncio.run(fxt_job_service.submit_train_job(fxt_job_payload))

        assert result.job_id == fxt_job.id
        fxt_job_repository.is_job_duplicate.assert_called_once_with(
            project_id=fxt_job_payload.project_id, payload=fxt_job_payload
        )
        fxt_job_repository.save.assert_called_once()

    def test_submit_train_job_duplicate(self, fxt_job_service, fxt_job_repository, fxt_job_payload):
        """Test job submission with duplicate job."""
        fxt_job_repository.is_job_duplicate.return_value = True

        with pytest.raises(DuplicateJobException):
            asyncio.run(fxt_job_service.submit_train_job(fxt_job_payload))

        fxt_job_repository.is_job_duplicate.assert_called_once_with(
            project_id=fxt_job_payload.project_id, payload=fxt_job_payload
        )

    def test_submit_train_job_integrity_error(self, fxt_job_service, fxt_job_repository, fxt_job_payload):
        """Test job submission with integrity error."""
        fxt_job_repository.is_job_duplicate.return_value = False
        fxt_job_repository.save.side_effect = IntegrityError("", "", "")

        with pytest.raises(ResourceNotFoundException) as exc_info:
            asyncio.run(fxt_job_service.submit_train_job(fxt_job_payload))

        # Check that the exception was raised with correct parameters
        assert str(fxt_job_payload.project_id) in str(exc_info.value)
        assert "project" in str(exc_info.value)

    def test_get_pending_train_job(self, fxt_job_service, fxt_job_repository, fxt_job):
        """Test getting pending training job."""
        fxt_job_repository.get_pending_job_by_type.return_value = fxt_job

        result = asyncio.run(fxt_job_service.get_pending_train_job())

        assert result == fxt_job
        fxt_job_repository.get_pending_job_by_type.assert_called_once_with(JobType.TRAINING)

    def test_get_pending_train_job_none(self, fxt_job_service, fxt_job_repository):
        """Test getting pending training job when none exists."""
        fxt_job_repository.get_pending_job_by_type.return_value = None

        result = asyncio.run(fxt_job_service.get_pending_train_job())

        assert result is None
        fxt_job_repository.get_pending_job_by_type.assert_called_once_with(JobType.TRAINING)

    def test_update_job_status_success(self, fxt_job_service, fxt_job_repository, fxt_job):
        """Test updating job status successfully."""
        fxt_job_repository.get_by_id.return_value = fxt_job
        fxt_job_repository.update.return_value = None

        asyncio.run(fxt_job_service.update_job_status(fxt_job.id, JobStatus.COMPLETED, "Test message"))

        assert fxt_job.status == JobStatus.COMPLETED
        assert fxt_job.message == "Test message"
        fxt_job_repository.get_by_id.assert_called_once_with(fxt_job.id)
        fxt_job_repository.update.assert_called_once_with(fxt_job)

    def test_update_job_status_without_message(self, fxt_job_service, fxt_job_repository, fxt_job):
        """Test updating job status without message."""
        fxt_job_repository.get_by_id.return_value = fxt_job
        fxt_job_repository.update.return_value = None
        original_message = fxt_job.message

        asyncio.run(fxt_job_service.update_job_status(fxt_job.id, JobStatus.COMPLETED))

        assert fxt_job.status == JobStatus.COMPLETED
        assert fxt_job.message == original_message
        fxt_job_repository.get_by_id.assert_called_once_with(fxt_job.id)
        fxt_job_repository.update.assert_called_once_with(fxt_job)

    def test_update_job_status_not_found(self, fxt_job_service, fxt_job_repository):
        """Test updating job status when job not found."""
        fxt_job_repository.get_by_id.return_value = None

        with pytest.raises(ResourceNotFoundException) as exc_info:
            asyncio.run(fxt_job_service.update_job_status("non-existent-id", JobStatus.COMPLETED))

        # Check that the exception was raised with correct parameters
        assert "non-existent-id" in str(exc_info.value)
        assert "job" in str(exc_info.value)
        fxt_job_repository.get_by_id.assert_called_once_with("non-existent-id")
