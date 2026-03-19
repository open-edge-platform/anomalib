# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from pydantic_models import StartupProjectSelectionSource
from pydantic_models.pipeline import Pipeline
from services import ProjectSelectionService, ResourceNotFoundError


@pytest.fixture(autouse=True)
def mock_db_context():
    with patch("services.project_selection_service.get_async_db_session_ctx") as mock_db_ctx:
        mock_session = AsyncMock()
        mock_db_ctx.return_value.__aenter__.return_value = mock_session
        mock_db_ctx.return_value.__aexit__.return_value = None
        yield mock_db_ctx


@pytest.fixture
def fxt_app_state_repository():
    return MagicMock()


@pytest.fixture
def fxt_project_repository():
    return MagicMock()


@pytest.fixture
def fxt_pipeline_repository():
    return MagicMock()


@pytest.fixture
def fxt_pipeline(fxt_project):
    return Pipeline(project_id=fxt_project.id)


class TestProjectSelectionService:
    def test_get_startup_project_selection_prefers_last_used_project(
        self,
        fxt_app_state_repository,
        fxt_project_repository,
        fxt_pipeline_repository,
        fxt_project,
    ):
        fxt_app_state_repository.get_last_used_project_id = AsyncMock(return_value=str(fxt_project.id))
        fxt_app_state_repository.clear_last_used_project_id = AsyncMock()
        fxt_project_repository.get_by_id = AsyncMock(return_value=fxt_project)
        fxt_pipeline_repository.get_active_pipeline = AsyncMock(return_value=None)

        with (
            patch("services.project_selection_service.AppStateRepository", return_value=fxt_app_state_repository),
            patch("services.project_selection_service.ProjectRepository", return_value=fxt_project_repository),
            patch("services.project_selection_service.PipelineRepository", return_value=fxt_pipeline_repository),
        ):
            result = asyncio.run(ProjectSelectionService.get_startup_project_selection())

        assert result.project_id == fxt_project.id
        assert result.source == StartupProjectSelectionSource.LAST_USED
        fxt_pipeline_repository.get_active_pipeline.assert_not_called()
        fxt_project_repository.get_all_pagination.assert_not_called()

    def test_get_startup_project_selection_falls_back_to_active_pipeline_when_last_used_is_missing(
        self,
        fxt_app_state_repository,
        fxt_project_repository,
        fxt_pipeline_repository,
        fxt_pipeline,
    ):
        fxt_app_state_repository.get_last_used_project_id = AsyncMock(return_value=None)
        fxt_app_state_repository.clear_last_used_project_id = AsyncMock()
        fxt_pipeline_repository.get_active_pipeline = AsyncMock(return_value=fxt_pipeline)

        with (
            patch("services.project_selection_service.AppStateRepository", return_value=fxt_app_state_repository),
            patch("services.project_selection_service.ProjectRepository", return_value=fxt_project_repository),
            patch("services.project_selection_service.PipelineRepository", return_value=fxt_pipeline_repository),
        ):
            result = asyncio.run(ProjectSelectionService.get_startup_project_selection())

        assert result.project_id == fxt_pipeline.project_id
        assert result.source == StartupProjectSelectionSource.ACTIVE_PIPELINE
        fxt_project_repository.get_all_pagination.assert_not_called()

    def test_get_startup_project_selection_falls_back_to_first_project(
        self,
        fxt_app_state_repository,
        fxt_project_repository,
        fxt_pipeline_repository,
        fxt_project,
    ):
        fxt_app_state_repository.get_last_used_project_id = AsyncMock(return_value=None)
        fxt_app_state_repository.clear_last_used_project_id = AsyncMock()
        fxt_pipeline_repository.get_active_pipeline = AsyncMock(return_value=None)
        fxt_project_repository.get_all_pagination = AsyncMock(return_value=[fxt_project])

        with (
            patch("services.project_selection_service.AppStateRepository", return_value=fxt_app_state_repository),
            patch("services.project_selection_service.ProjectRepository", return_value=fxt_project_repository),
            patch("services.project_selection_service.PipelineRepository", return_value=fxt_pipeline_repository),
        ):
            result = asyncio.run(ProjectSelectionService.get_startup_project_selection())

        assert result.project_id == fxt_project.id
        assert result.source == StartupProjectSelectionSource.FIRST_PROJECT

    def test_get_startup_project_selection_returns_none_when_no_project_exists(
        self,
        fxt_app_state_repository,
        fxt_project_repository,
        fxt_pipeline_repository,
    ):
        fxt_app_state_repository.get_last_used_project_id = AsyncMock(return_value=None)
        fxt_app_state_repository.clear_last_used_project_id = AsyncMock()
        fxt_pipeline_repository.get_active_pipeline = AsyncMock(return_value=None)
        fxt_project_repository.get_all_pagination = AsyncMock(return_value=[])

        with (
            patch("services.project_selection_service.AppStateRepository", return_value=fxt_app_state_repository),
            patch("services.project_selection_service.ProjectRepository", return_value=fxt_project_repository),
            patch("services.project_selection_service.PipelineRepository", return_value=fxt_pipeline_repository),
        ):
            result = asyncio.run(ProjectSelectionService.get_startup_project_selection())

        assert result.project_id is None
        assert result.source == StartupProjectSelectionSource.NONE

    def test_get_startup_project_selection_clears_invalid_last_used_project_and_uses_active_pipeline(
        self,
        fxt_app_state_repository,
        fxt_project_repository,
        fxt_pipeline_repository,
        fxt_pipeline,
    ):
        missing_project_id = str(uuid4())
        fxt_app_state_repository.get_last_used_project_id = AsyncMock(return_value=missing_project_id)
        fxt_app_state_repository.clear_last_used_project_id = AsyncMock()
        fxt_project_repository.get_by_id = AsyncMock(return_value=None)
        fxt_pipeline_repository.get_active_pipeline = AsyncMock(return_value=fxt_pipeline)

        with (
            patch("services.project_selection_service.AppStateRepository", return_value=fxt_app_state_repository),
            patch("services.project_selection_service.ProjectRepository", return_value=fxt_project_repository),
            patch("services.project_selection_service.PipelineRepository", return_value=fxt_pipeline_repository),
        ):
            result = asyncio.run(ProjectSelectionService.get_startup_project_selection())

        assert result.project_id == fxt_pipeline.project_id
        assert result.source == StartupProjectSelectionSource.ACTIVE_PIPELINE
        fxt_app_state_repository.clear_last_used_project_id.assert_called_once_with()

    def test_set_last_used_project_persists_project_id(
        self, fxt_app_state_repository, fxt_project_repository, fxt_project
    ):
        fxt_project_repository.get_by_id = AsyncMock(return_value=fxt_project)
        fxt_app_state_repository.set_last_used_project_id = AsyncMock()

        with (
            patch("services.project_selection_service.AppStateRepository", return_value=fxt_app_state_repository),
            patch("services.project_selection_service.ProjectRepository", return_value=fxt_project_repository),
        ):
            asyncio.run(ProjectSelectionService.set_last_used_project(fxt_project.id))

        fxt_project_repository.get_by_id.assert_called_once_with(fxt_project.id)
        fxt_app_state_repository.set_last_used_project_id.assert_called_once_with(str(fxt_project.id))

    def test_set_last_used_project_raises_when_project_does_not_exist(
        self,
        fxt_app_state_repository,
        fxt_project_repository,
    ):
        project_id = uuid4()
        fxt_project_repository.get_by_id = AsyncMock(return_value=None)
        fxt_app_state_repository.set_last_used_project_id = AsyncMock()

        with (
            patch("services.project_selection_service.AppStateRepository", return_value=fxt_app_state_repository),
            patch("services.project_selection_service.ProjectRepository", return_value=fxt_project_repository),
            pytest.raises(ResourceNotFoundError),
        ):
            asyncio.run(ProjectSelectionService.set_last_used_project(project_id))

        fxt_app_state_repository.set_last_used_project_id.assert_not_called()
