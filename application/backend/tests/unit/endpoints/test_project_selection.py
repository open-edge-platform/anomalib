# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import status

from api.dependencies import get_project_selection_service
from main import app
from pydantic_models import StartupProjectSelection, StartupProjectSelectionSource
from services import ProjectSelectionService, ResourceNotFoundError
from services.exceptions import ResourceType
from utils.short_uuid import ShortUUID


@pytest.fixture
def fxt_project_selection_service() -> MagicMock:
    project_selection_service = MagicMock(spec=ProjectSelectionService)
    project_selection_service.get_startup_project_selection = AsyncMock()
    project_selection_service.set_last_used_project = AsyncMock()
    app.dependency_overrides[get_project_selection_service] = lambda: project_selection_service
    return project_selection_service


def test_get_startup_project_selection(fxt_client, fxt_project_selection_service, fxt_project):
    fxt_project_selection_service.get_startup_project_selection.return_value = StartupProjectSelection(
        project_id=fxt_project.id,
        source=StartupProjectSelectionSource.LAST_USED,
    )

    response = fxt_client.get("/api/projects/startup-selection")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "project_id": str(fxt_project.id),
        "source": StartupProjectSelectionSource.LAST_USED,
    }
    fxt_project_selection_service.get_startup_project_selection.assert_called_once_with()


def test_update_last_used_project(fxt_client, fxt_project_selection_service):
    project_id = ShortUUID.generate()

    response = fxt_client.put("/api/projects/last-used", json={"project_id": str(project_id)})

    assert response.status_code == status.HTTP_204_NO_CONTENT
    fxt_project_selection_service.set_last_used_project.assert_called_once_with(project_id)


def test_update_last_used_project_not_found(fxt_client, fxt_project_selection_service):
    project_id = ShortUUID.generate()
    fxt_project_selection_service.set_last_used_project.side_effect = ResourceNotFoundError(
        resource_type=ResourceType.PROJECT,
        resource_id=str(project_id),
    )

    response = fxt_client.put("/api/projects/last-used", json={"project_id": str(project_id)})

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"].lower()
