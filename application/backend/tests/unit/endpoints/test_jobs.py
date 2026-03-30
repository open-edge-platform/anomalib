# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest
from fastapi import status
from sse_starlette import ServerSentEvent

from api.dependencies import get_job_service
from exceptions import JobNotDeletableException, ResourceNotFoundException
from main import app
from pydantic_models import JobList
from pydantic_models.base import Pagination
from pydantic_models.job import JobStatus
from services import JobService


@pytest.fixture
def fxt_job_service() -> MagicMock:
    job_service = MagicMock(spec=JobService)
    app.dependency_overrides[get_job_service] = lambda: job_service
    return job_service


def test_get_jobs(fxt_client, fxt_job_service, fxt_job):
    fxt_job_service.get_job_list.return_value = JobList(
        jobs=[fxt_job],
        pagination=Pagination(offset=0, limit=20, count=1, total=1),
    )

    response = fxt_client.get("/api/jobs")
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()["jobs"]) == 1
    fxt_job_service.get_job_list.assert_called_once_with(limit=20, offset=0)


def test_get_jobs_empty(fxt_client, fxt_job_service, fxt_job):
    fxt_job_service.get_job_list.return_value = JobList(
        jobs=[],
        pagination=Pagination(offset=0, limit=20, count=0, total=0),
    )

    response = fxt_client.get("/api/jobs")
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()["jobs"]) == 0
    fxt_job_service.get_job_list.assert_called_once_with(limit=20, offset=0)


def test_get_job_logs_success(fxt_client, fxt_job_service, fxt_job):
    """Test successful log streaming endpoint."""

    # Mock the stream_logs generator
    async def mock_stream():
        yield ServerSentEvent(data='{"level": "INFO", "message": "Line 1"}')
        yield ServerSentEvent(data='{"level": "INFO", "message": "Line 2"}')

    fxt_job_service.stream_logs.return_value = mock_stream()

    response = fxt_client.get(f"/api/jobs/{fxt_job.id}/logs")
    assert response.status_code == status.HTTP_200_OK

    # Verify the streamed content
    content = response.content.decode("utf-8")
    lines = [line for line in content.split("\n") if line]
    assert len(lines) == 4  # 2 events + 2 newlines
    assert '"level": "INFO"' in lines[0]
    assert '"message": "Line 1"' in lines[0]

    fxt_job_service.stream_logs.assert_called_once_with(job_id=fxt_job.id)


def test_get_job_logs_invalid_uuid(fxt_client, fxt_job_service):
    """Test log streaming with invalid job ID."""
    response = fxt_client.get("/api/jobs/invalid-uuid/logs")
    # FastAPI returns 400 for invalid UUID in path parameter
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_delete_job_failed(fxt_client, fxt_job_service, fxt_job):
    """DELETE /api/jobs/{job_id} returns 204 for a failed job."""
    fxt_job_service.delete_job.return_value = None

    response = fxt_client.delete(f"/api/jobs/{fxt_job.id}")

    assert response.status_code == status.HTTP_204_NO_CONTENT
    fxt_job_service.delete_job.assert_called_once_with(job_id=fxt_job.id)


def test_delete_job_canceled(fxt_client, fxt_job_service, fxt_job):
    """DELETE /api/jobs/{job_id} returns 204 for a canceled job."""
    fxt_job_service.delete_job.return_value = None

    response = fxt_client.delete(f"/api/jobs/{fxt_job.id}")

    assert response.status_code == status.HTTP_204_NO_CONTENT
    fxt_job_service.delete_job.assert_called_once_with(job_id=fxt_job.id)


def test_delete_job_running_returns_409(fxt_client, fxt_job_service, fxt_job):
    """DELETE /api/jobs/{job_id} returns 409 when the job is running."""
    fxt_job_service.delete_job.side_effect = JobNotDeletableException(job_id=fxt_job.id, job_status=JobStatus.RUNNING)

    response = fxt_client.delete(f"/api/jobs/{fxt_job.id}")

    assert response.status_code == status.HTTP_409_CONFLICT
    assert response.json()["error_code"] == "job_not_deletable"


def test_delete_job_pending_returns_409(fxt_client, fxt_job_service, fxt_job):
    """DELETE /api/jobs/{job_id} returns 409 when the job is pending."""
    fxt_job_service.delete_job.side_effect = JobNotDeletableException(job_id=fxt_job.id, job_status=JobStatus.PENDING)

    response = fxt_client.delete(f"/api/jobs/{fxt_job.id}")

    assert response.status_code == status.HTTP_409_CONFLICT
    assert response.json()["error_code"] == "job_not_deletable"


def test_delete_job_not_found_returns_404(fxt_client, fxt_job_service, fxt_job):
    """DELETE /api/jobs/{job_id} returns 404 when the job does not exist."""
    fxt_job_service.delete_job.side_effect = ResourceNotFoundException(resource_id=fxt_job.id, resource_name="job")

    response = fxt_client.delete(f"/api/jobs/{fxt_job.id}")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["error_code"] == "job_not_found"
