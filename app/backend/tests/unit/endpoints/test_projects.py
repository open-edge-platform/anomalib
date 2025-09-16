# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import status

from api.dependencies import get_project_service
from main import app
from models import ProjectList
from services import ProjectService


@pytest.fixture
def fxt_project_service() -> MagicMock:
    project_service = MagicMock(spec=ProjectService)
    app.dependency_overrides[get_project_service] = lambda: project_service
    return project_service


def test_get_projects(fxt_client, fxt_project_service, fxt_project):
    fxt_project_service.get_project_list.return_value = ProjectList(projects=[fxt_project])

    response = fxt_client.get("/api/projects")
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()["projects"]) == 1
    fxt_project_service.get_project_list.assert_called_once()


def test_get_project_not_found(fxt_client, fxt_project_service):
    project_id = uuid4()
    fxt_project_service.get_project_by_id.return_value = None

    response = fxt_client.get(f"/api/projects/{project_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    fxt_project_service.get_project_by_id.assert_called_once_with(project_id)
