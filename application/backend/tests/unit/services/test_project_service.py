# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from repositories import ProjectRepository
from services import ProjectService


@pytest.fixture
def fxt_project_repository():
    """Fixture for a mock project repository."""
    return MagicMock(spec=ProjectRepository)


@pytest.fixture
def fxt_project_service():
    """Fixture for ProjectService - all methods are static."""
    return ProjectService


@pytest.fixture(autouse=True)
def mock_db_context():
    """Mock the database context for all tests."""
    with patch("services.project_service.get_async_db_session_ctx") as mock_db_ctx:
        mock_session = AsyncMock()
        mock_db_ctx.return_value.__aenter__.return_value = mock_session
        mock_db_ctx.return_value.__aexit__.return_value = None
        yield mock_db_ctx


class TestProjectService:
    def test_get_project_list(self, fxt_project_service, fxt_project_repository, fxt_project_list):
        """Test getting project list."""
        fxt_project_repository.get_all_count = AsyncMock(return_value=len(fxt_project_list.projects))
        fxt_project_repository.get_all_pagination = AsyncMock(return_value=fxt_project_list.projects)

        with patch("services.project_service.ProjectRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_project_repository

            result = asyncio.run(fxt_project_service.get_project_list(limit=20, offset=0))

        assert result.projects == fxt_project_list.projects
        assert result.pagination.total == len(fxt_project_list.projects)
        fxt_project_repository.get_all_count.assert_called_once()
        fxt_project_repository.get_all_pagination.assert_called_once_with(limit=20, offset=0)

    def test_get_project_by_id(self, fxt_project_service, fxt_project_repository, fxt_project):
        """Test getting project by ID."""
        fxt_project_repository.get_by_id.return_value = fxt_project

        with patch("services.project_service.ProjectRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_project_repository

            result = asyncio.run(fxt_project_service.get_project_by_id(fxt_project.id))

        assert result == fxt_project
        fxt_project_repository.get_by_id.assert_called_once_with(fxt_project.id)

    def test_get_project_by_id_not_found(self, fxt_project_service, fxt_project_repository):
        """Test getting project by ID when not found."""
        fxt_project_repository.get_by_id.return_value = None

        with patch("services.project_service.ProjectRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_project_repository

            result = asyncio.run(fxt_project_service.get_project_by_id("non-existent-id"))

        assert result is None
        fxt_project_repository.get_by_id.assert_called_once_with("non-existent-id")

    def test_create_project(self, fxt_project_service, fxt_project_repository, fxt_project):
        """Test creating a project."""
        fxt_project_repository.save.return_value = fxt_project

        with patch("services.project_service.ProjectRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_project_repository

            result = asyncio.run(fxt_project_service.create_project(fxt_project))

        assert result == fxt_project
        fxt_project_repository.save.assert_called_once_with(fxt_project)

    @patch("services.project_service.PipelineService")
    @patch("services.project_service.ModelService")
    @patch("services.project_service.DatasetSnapshotService")
    @patch("services.project_service.MediaService")
    @patch("services.project_service.JobService")
    @patch("services.project_service.ConfigurationService")
    def test_delete_project(
        self,
        mock_config_service,
        mock_job_service,
        mock_media_service,
        mock_snapshot_service,
        mock_model_service,
        mock_pipeline_service,
        fxt_project_service,
        fxt_project_repository,
        fxt_project,
    ):
        """Test deleting a project."""
        fxt_project_repository.delete_by_id.return_value = None

        # Configure mocks to be awaitable
        mock_pipeline_service.delete_project_pipelines_db = AsyncMock()
        mock_model_service.delete_project_models_db = AsyncMock()
        mock_snapshot_service.delete_project_snapshots_db = AsyncMock()
        mock_media_service.delete_project_media_db = AsyncMock()
        mock_job_service.delete_project_jobs_db = AsyncMock()
        mock_config_service.delete_project_source_db = AsyncMock()
        mock_config_service.delete_project_sink_db = AsyncMock()

        mock_model_service.cleanup_project_model_files = AsyncMock()
        mock_snapshot_service.cleanup_project_snapshot_files = AsyncMock()
        mock_media_service.cleanup_project_media_files = AsyncMock()

        with patch("services.project_service.ProjectRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_project_repository

            asyncio.run(fxt_project_service.delete_project(fxt_project.id))

        # Verify DB deletions
        mock_pipeline_service.delete_project_pipelines_db.assert_called_once()
        mock_model_service.delete_project_models_db.assert_called_once()
        mock_snapshot_service.delete_project_snapshots_db.assert_called_once()
        mock_media_service.delete_project_media_db.assert_called_once()
        mock_job_service.delete_project_jobs_db.assert_called_once()
        mock_config_service.delete_project_source_db.assert_called_once()
        mock_config_service.delete_project_sink_db.assert_called_once()
        fxt_project_repository.delete_by_id.assert_called_once_with(fxt_project.id)

        # Verify File cleanup
        mock_model_service.cleanup_project_model_files.assert_called_once_with(fxt_project.id)
        mock_snapshot_service.cleanup_project_snapshot_files.assert_called_once_with(fxt_project.id)
        mock_media_service.cleanup_project_media_files.assert_called_once_with(fxt_project.id)

    @patch("services.project_service.PipelineService")
    @patch("services.project_service.ModelService")
    @patch("services.project_service.DatasetSnapshotService")
    @patch("services.project_service.MediaService")
    @patch("services.project_service.JobService")
    @patch("services.project_service.ConfigurationService")
    def test_delete_project_db_failure(
        self,
        mock_config_service,
        mock_job_service,
        mock_media_service,
        mock_snapshot_service,
        mock_model_service,
        mock_pipeline_service,
        fxt_project_service,
        fxt_project_repository,
        fxt_project,
    ):
        """Test deleting a project with DB failure."""
        mock_pipeline_service.delete_project_pipelines_db = AsyncMock()
        mock_pipeline_service.delete_project_pipelines_db.side_effect = Exception("DB Error")

        mock_model_service.cleanup_project_model_files = AsyncMock()
        mock_snapshot_service.cleanup_project_snapshot_files = AsyncMock()
        mock_media_service.cleanup_project_media_files = AsyncMock()

        with patch("services.project_service.ProjectRepository") as mock_repo_class:
            mock_repo_class.return_value = fxt_project_repository

            with pytest.raises(RuntimeError, match="Failed to delete project"):
                asyncio.run(fxt_project_service.delete_project(fxt_project.id))

        # Verify DB deletions attempted
        mock_pipeline_service.delete_project_pipelines_db.assert_called_once()

        # Verify File cleanup NOT called
        mock_model_service.cleanup_project_model_files.assert_not_called()
        mock_snapshot_service.cleanup_project_snapshot_files.assert_not_called()
        mock_media_service.cleanup_project_media_files.assert_not_called()
