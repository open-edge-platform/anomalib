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
