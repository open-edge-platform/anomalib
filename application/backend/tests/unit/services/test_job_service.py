# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from freezegun import freeze_time
from sqlalchemy.exc import IntegrityError

from exceptions import DuplicateJobException, ResourceNotFoundException
from pydantic_models import JobStatus, JobType
from repositories import JobRepository
from services import JobService


@pytest.fixture
def fxt_job_repository():
    """Fixture for a mock job repository."""
    mock_repo = MagicMock(spec=JobRepository)
    # Set up async methods to return coroutines
    mock_repo.get_by_id = AsyncMock()
    mock_repo.get_one = AsyncMock()
    mock_repo.get_all = AsyncMock()
    mock_repo.get_all_count = AsyncMock()
    mock_repo.get_all_pagination = AsyncMock()
    mock_repo.save = AsyncMock()
    mock_repo.update = AsyncMock()
    mock_repo.delete_by_id = AsyncMock()
    mock_repo.get_pending_job_by_type = AsyncMock()
    return mock_repo


@pytest.fixture(autouse=True)
def mock_db_context():
    """Mock the database context for all tests."""
    with patch("services.job_service.get_async_db_session_ctx") as mock_db_ctx:
        mock_session = AsyncMock()
        mock_db_ctx.return_value.__aenter__.return_value = mock_session
        mock_db_ctx.return_value.__aexit__.return_value = None
        yield mock_db_ctx


class TestJobService:
    def test_get_job_list(self, fxt_job_repository, fxt_job_list):
        """Test getting job list."""
        fxt_job_repository.get_all_count.return_value = len(fxt_job_list.jobs)
        fxt_job_repository.get_all_pagination.return_value = fxt_job_list.jobs

        with patch("services.job_service.JobRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_job_repository

            result = asyncio.run(JobService.get_job_list(limit=20, offset=0))

        assert result.jobs == fxt_job_list.jobs
        assert result.pagination.total == len(fxt_job_list.jobs)
        fxt_job_repository.get_all_count.assert_called_once()
        fxt_job_repository.get_all_pagination.assert_called_once_with(limit=20, offset=0, extra_filters=None)

    @pytest.mark.parametrize(
        "job_exists,expected_result",
        [
            (True, "job_found"),
            (False, None),
        ],
    )
    def test_get_job_by_id(self, fxt_job_repository, fxt_job, job_exists, expected_result):
        """Test getting job by ID with different scenarios."""
        if job_exists:
            fxt_job_repository.get_by_id.return_value = fxt_job
            job_id = fxt_job.id
        else:
            fxt_job_repository.get_by_id.return_value = None
            job_id = "non-existent-id"

        with patch("services.job_service.JobRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_job_repository

            result = asyncio.run(JobService.get_job_by_id(job_id))

        if job_exists:
            assert result == fxt_job
        else:
            assert result is None
        fxt_job_repository.get_by_id.assert_called_once_with(job_id)

    @pytest.mark.parametrize(
        "is_duplicate,save_side_effect,expected_exception,expected_message",
        [
            (False, None, None, "success"),
            (True, None, DuplicateJobException, None),
            (
                False,
                IntegrityError("", {}, ValueError("Simulated database error")),
                ResourceNotFoundException,
                "project",
            ),
        ],
    )
    def test_submit_train_job(
        self,
        fxt_job_repository,
        fxt_job_payload,
        fxt_job,
        is_duplicate,
        save_side_effect,
        expected_exception,
        expected_message,
    ):
        """Test job submission with different scenarios."""
        fxt_job_repository.is_job_duplicate.return_value = is_duplicate
        if save_side_effect:
            fxt_job_repository.save.side_effect = save_side_effect
        else:
            fxt_job_repository.save.return_value = fxt_job

        with patch("services.job_service.JobRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_job_repository

            if expected_exception:
                with pytest.raises(expected_exception) as exc_info:
                    asyncio.run(JobService().submit_train_job(fxt_job_payload))

                if expected_message:
                    assert expected_message in str(exc_info.value)
            else:
                result = asyncio.run(JobService().submit_train_job(fxt_job_payload))
                assert result.job_id == fxt_job.id

        fxt_job_repository.is_job_duplicate.assert_called_once_with(
            project_id=fxt_job_payload.project_id,
            payload=fxt_job_payload,
        )
        if not is_duplicate:
            fxt_job_repository.save.assert_called_once()

    @pytest.mark.parametrize(
        "job_exists,expected_result",
        [
            (True, "job_found"),
            (False, None),
        ],
    )
    def test_get_pending_train_job(self, fxt_job_repository, fxt_job, job_exists, expected_result):
        """Test getting pending training job with different scenarios."""
        if job_exists:
            fxt_job_repository.get_pending_job_by_type.return_value = fxt_job
        else:
            fxt_job_repository.get_pending_job_by_type.return_value = None

        with patch("services.job_service.JobRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_job_repository

            result = asyncio.run(JobService.get_pending_train_job())

        if job_exists:
            assert result == fxt_job
        else:
            assert result is None
        fxt_job_repository.get_pending_job_by_type.assert_called_once_with(JobType.TRAINING)

    @pytest.mark.parametrize(
        "has_message,message",
        [
            (True, "Test message"),
            (False, None),
        ],
    )
    @freeze_time("2025-01-01 00:00:00")
    def test_update_job_status_success(self, fxt_job_repository, fxt_job, has_message, message):
        """Test updating job status successfully with and without message."""
        # Expected updates include end_time since status is COMPLETED
        frozen_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        expected_updates = {
            "status": JobStatus.COMPLETED,
            "end_time": frozen_time,
            "progress": 100,
        }
        if has_message:
            expected_updates["message"] = message

        # Create an updated job object that the repository would return
        updated_job = fxt_job.model_copy(update=expected_updates)
        fxt_job_repository.get_by_id.return_value = fxt_job
        fxt_job_repository.update.return_value = updated_job

        with patch("services.job_service.JobRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_job_repository

            if has_message:
                asyncio.run(JobService.update_job_status(fxt_job.id, JobStatus.COMPLETED, message))
            else:
                asyncio.run(JobService.update_job_status(fxt_job.id, JobStatus.COMPLETED))

        fxt_job_repository.get_by_id.assert_called_once_with(fxt_job.id)
        fxt_job_repository.update.assert_called_once_with(fxt_job, expected_updates)

    def test_update_job_status_not_found(self, fxt_job_repository):
        """Test updating job status when job not found."""
        fxt_job_repository.get_by_id.return_value = None

        with patch("services.job_service.JobRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_job_repository

            with pytest.raises(ResourceNotFoundException) as exc_info:
                asyncio.run(JobService.update_job_status("non-existent-id", JobStatus.COMPLETED))

        # Check that the exception was raised with correct parameters
        assert "non-existent-id" in str(exc_info.value)
        assert "job" in str(exc_info.value)
        fxt_job_repository.get_by_id.assert_called_once_with("non-existent-id")

    def test_stream_logs_file_not_found(self, fxt_job):
        """Test streaming logs when log file doesn't exist."""
        with (
            patch("core.logging.utils.get_job_logs_path") as mock_get_path,
            patch("services.job_service.anyio.Path.exists") as mock_exists,
        ):
            mock_get_path.return_value = "/fake/path/job.log"
            mock_exists.return_value = False

            with pytest.raises(ResourceNotFoundException) as exc_info:

                async def consume_stream():
                    async for _ in JobService.stream_logs(fxt_job.id):
                        pass

                asyncio.run(consume_stream())

            assert "job_logs" in str(exc_info.value)

    def test_stream_logs_success(self, fxt_job_repository, fxt_job):
        """Test streaming logs successfully from a completed job."""
        log_lines = ['{"level": "INFO", "message": "Line 1"}', '{"level": "INFO", "message": "Line 2"}']

        # Mock job as completed
        completed_job = fxt_job.model_copy(update={"status": JobStatus.COMPLETED})
        fxt_job_repository.get_by_id.return_value = completed_job

        with (
            patch("core.logging.utils.get_job_logs_path") as mock_get_path,
            patch("services.job_service.anyio.Path.exists") as mock_exists,
            patch("services.job_service.anyio.open_file") as mock_open_file,
            patch("services.job_service.JobRepository") as mock_repo_class,
        ):
            mock_get_path.return_value = "/fake/path/job.log"
            mock_exists.return_value = True
            mock_repo_class.return_value = fxt_job_repository

            # Mock file with async readline method
            mock_file = MagicMock()
            mock_file.readline = AsyncMock(side_effect=[*log_lines, ""])  # Empty string signals EOF

            # Create an async context manager
            async_cm = MagicMock()
            async_cm.__aenter__ = AsyncMock(return_value=mock_file)
            async_cm.__aexit__ = AsyncMock(return_value=None)

            # anyio.open_file() is a coroutine that returns an async context manager when awaited
            async def mock_anyio_open_file(*args, **kwargs):
                return async_cm

            mock_open_file.side_effect = mock_anyio_open_file

            async def consume_stream():
                result = []
                async for line in JobService.stream_logs(fxt_job.id):
                    result.append(line.data)
                return result

            result = asyncio.run(consume_stream())

            assert len(result) == 2
            assert result == log_lines
