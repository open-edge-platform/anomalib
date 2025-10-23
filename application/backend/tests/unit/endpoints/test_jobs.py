# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest
from fastapi import status

from api.dependencies import get_job_service
from main import app
from pydantic_models import JobList
from services import JobService


@pytest.fixture
def fxt_job_service() -> MagicMock:
    job_service = MagicMock(spec=JobService)
    app.dependency_overrides[get_job_service] = lambda: job_service
    return job_service


def test_get_jobs(fxt_client, fxt_job_service, fxt_job):
    fxt_job_service.get_job_list.return_value = JobList(jobs=[fxt_job])

    response = fxt_client.get("/api/jobs")
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()["jobs"]) == 1
    fxt_job_service.get_job_list.assert_called_once()


def test_get_jobs_empty(fxt_client, fxt_job_service, fxt_job):
    fxt_job_service.get_job_list.return_value = JobList(jobs=[])

    response = fxt_client.get("/api/jobs")
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()["jobs"]) == 0
    fxt_job_service.get_job_list.assert_called_once()


def test_get_job_logs_success(fxt_client, fxt_job_service, fxt_job):
    """Test successful log streaming endpoint."""

    # Mock the stream_logs generator
    async def mock_stream():
        yield '{"level": "INFO", "message": "Line 1"}\n'
        yield '{"level": "INFO", "message": "Line 2"}\n'

    fxt_job_service.stream_logs.return_value = mock_stream()

    response = fxt_client.get(f"/api/jobs/{fxt_job.id}/logs")
    assert response.status_code == status.HTTP_200_OK

    # Verify the streamed content
    content = response.content.decode("utf-8")
    lines = [line for line in content.split("\n") if line]
    assert len(lines) == 2
    assert '"level": "INFO"' in lines[0]
    assert '"message": "Line 1"' in lines[0]

    fxt_job_service.stream_logs.assert_called_once_with(job_id=fxt_job.id)


def test_get_job_logs_invalid_uuid(fxt_client, fxt_job_service):
    """Test log streaming with invalid job ID."""
    response = fxt_client.get("/api/jobs/invalid-uuid/logs")
    # FastAPI returns 400 for invalid UUID in path parameter
    assert response.status_code == status.HTTP_400_BAD_REQUEST
