# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import MagicMock

import pytest

from repositories import ProjectRepository
from services import ProjectService


@pytest.fixture
def fxt_project_repository():
    """Fixture for a mock project repository."""
    return MagicMock(spec=ProjectRepository)


@pytest.fixture
def fxt_project_service(fxt_db_session, fxt_project_repository):
    """Fixture for ProjectService with mocked repository."""
    service = ProjectService(fxt_db_session)
    service.project_repository = fxt_project_repository
    return service


class TestProjectService:
    def test_get_project_list(self, fxt_project_service, fxt_project_repository, fxt_project_list):
        """Test getting project list."""
        fxt_project_repository.get_all.return_value = fxt_project_list.projects

        result = asyncio.run(fxt_project_service.get_project_list())

        assert result == fxt_project_list
        fxt_project_repository.get_all.assert_called_once()

    def test_get_project_by_id(self, fxt_project_service, fxt_project_repository, fxt_project):
        """Test getting project by ID."""
        fxt_project_repository.get_by_id.return_value = fxt_project

        result = asyncio.run(fxt_project_service.get_project_by_id(fxt_project.id))

        assert result == fxt_project
        fxt_project_repository.get_by_id.assert_called_once_with(fxt_project.id)

    def test_get_project_by_id_not_found(self, fxt_project_service, fxt_project_repository):
        """Test getting project by ID when not found."""
        fxt_project_repository.get_by_id.return_value = None

        result = asyncio.run(fxt_project_service.get_project_by_id("non-existent-id"))

        assert result is None
        fxt_project_repository.get_by_id.assert_called_once_with("non-existent-id")

    def test_create_project(self, fxt_project_service, fxt_project_repository, fxt_project):
        """Test creating a project."""
        fxt_project_repository.save.return_value = fxt_project

        result = asyncio.run(fxt_project_service.create_project(fxt_project))

        assert result == fxt_project
        fxt_project_repository.save.assert_called_once_with(fxt_project)

    def test_delete_project(self, fxt_project_service, fxt_project_repository, fxt_project):
        """Test deleting a project."""
        fxt_project_repository.delete_by_id.return_value = None

        asyncio.run(fxt_project_service.delete_project(fxt_project.id))

        fxt_project_repository.delete_by_id.assert_called_once_with(fxt_project.id)
