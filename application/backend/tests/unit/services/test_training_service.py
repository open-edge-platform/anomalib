# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_models import JobStatus
from repositories.binary_repo import ImageBinaryRepository, ModelBinaryRepository
from services import TrainingService


@pytest.fixture
def fxt_model_binary_repo():
    """Fixture for a mock model binary repository."""
    return MagicMock(spec=ModelBinaryRepository)


@pytest.fixture
def fxt_image_binary_repo():
    """Fixture for a mock image binary repository."""
    return MagicMock(spec=ImageBinaryRepository)


@pytest.fixture
def fxt_mock_job_service():
    """Fixture for a mock job service instance."""
    mock_job_service = MagicMock()
    mock_job_service.get_pending_train_job = AsyncMock()
    mock_job_service.update_job_status = AsyncMock()
    return mock_job_service


@pytest.fixture
def fxt_mock_model_service():
    """Fixture for a mock model service instance."""
    mock_model_service = MagicMock()
    mock_model_service.create_model = AsyncMock()
    mock_model_service.delete_model = AsyncMock()
    return mock_model_service


@pytest.fixture
def fxt_mock_binary_repos(fxt_model_binary_repo, fxt_image_binary_repo):
    """Fixture for mock binary repositories."""
    with (
        patch("services.training_service.ModelBinaryRepository") as mock_model_bin_repo_class,
        patch("services.training_service.ImageBinaryRepository") as mock_image_bin_repo_class,
    ):
        mock_model_bin_repo_class.return_value = fxt_model_binary_repo
        mock_image_bin_repo_class.return_value = fxt_image_binary_repo
        yield mock_model_bin_repo_class, mock_image_bin_repo_class


@pytest.fixture
def fxt_mock_anomalib_components():
    """Fixture for mock Anomalib components."""
    with (
        patch("services.training_service.Folder") as mock_folder_class,
        patch("services.training_service.get_model") as mock_get_model,
        patch("services.training_service.Engine") as mock_engine_class,
    ):
        mock_folder = MagicMock()
        mock_folder_class.return_value = mock_folder

        mock_anomalib_model = MagicMock()
        mock_get_model.return_value = mock_anomalib_model

        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine
        mock_engine.export.return_value = "/path/to/exported/model"

        yield {
            "folder_class": mock_folder_class,
            "folder": mock_folder,
            "get_model": mock_get_model,
            "anomalib_model": mock_anomalib_model,
            "engine_class": mock_engine_class,
            "engine": mock_engine,
        }


@pytest.fixture
def fxt_mock_job_service_class(fxt_mock_job_service):
    """Fixture that patches JobService class."""
    with patch("services.training_service.JobService") as mock_job_service_class:
        mock_job_service_class.return_value = fxt_mock_job_service
        yield mock_job_service_class


@pytest.fixture
def fxt_mock_model_service_class(fxt_mock_model_service):
    """Fixture that patches ModelService class."""
    with patch("services.training_service.ModelService") as mock_model_service_class:
        mock_model_service_class.return_value = fxt_mock_model_service
        yield mock_model_service_class


class TestTrainingService:
    def test_train_pending_job_no_pending_jobs(self, fxt_mock_job_service_class, fxt_mock_job_service):
        """Test training when no pending jobs exist."""
        fxt_mock_job_service.get_pending_train_job.return_value = None

        result = asyncio.run(TrainingService.train_pending_job())

        assert result is None
        fxt_mock_job_service.get_pending_train_job.assert_called_once()

    def test_train_pending_job_success(
        self,
        fxt_job,
        fxt_model,
        fxt_mock_job_service_class,
        fxt_mock_model_service_class,
        fxt_mock_job_service,
        fxt_mock_model_service,
    ):
        """Test successful training of a pending job."""
        fxt_job.payload = {"model_name": "padim"}
        fxt_mock_job_service.get_pending_train_job.return_value = fxt_job
        fxt_mock_model_service.create_model.return_value = fxt_model

        with patch("services.training_service.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = fxt_model

            result = asyncio.run(TrainingService.train_pending_job())

            assert result == fxt_model
            # Should be called twice: RUNNING then COMPLETED
            assert fxt_mock_job_service.update_job_status.call_count == 2
            fxt_mock_job_service.update_job_status.assert_any_call(
                job_id=fxt_job.id, status=JobStatus.RUNNING, message="Training started"
            )
            fxt_mock_job_service.update_job_status.assert_any_call(
                job_id=fxt_job.id, status=JobStatus.COMPLETED, message="Training completed successfully"
            )
            fxt_mock_model_service.create_model.assert_called_once()

    @pytest.mark.parametrize(
        "exception,expected_message",
        [
            (Exception("Training failed"), "Training failed"),
            (ValueError("Training failed - model is None"), "Training failed - model is None"),
        ],
    )
    def test_train_pending_job_training_failures(
        self,
        fxt_job,
        fxt_mock_job_service_class,
        fxt_mock_model_service_class,
        fxt_mock_job_service,
        exception,
        expected_message,
    ):
        """Test training failure handling with different failure scenarios."""
        fxt_job.payload = {"model_name": "padim"}
        fxt_mock_job_service.get_pending_train_job.return_value = fxt_job

        with patch("services.training_service.asyncio.to_thread") as mock_to_thread:
            if isinstance(exception, ValueError) and "model is None" in str(exception):
                mock_to_thread.return_value = None
            else:
                mock_to_thread.side_effect = exception

            with pytest.raises(type(exception), match=expected_message):
                asyncio.run(TrainingService.train_pending_job())

            # Should be called twice: RUNNING then FAILED
            assert fxt_mock_job_service.update_job_status.call_count == 2
            fxt_mock_job_service.update_job_status.assert_any_call(
                job_id=fxt_job.id, status=JobStatus.RUNNING, message="Training started"
            )
            fxt_mock_job_service.update_job_status.assert_any_call(
                job_id=fxt_job.id, status=JobStatus.FAILED, message=f"Failed with exception: {expected_message}"
            )

    def test_train_pending_job_cleanup_on_failure(
        self,
        fxt_job,
        fxt_model,
        fxt_mock_job_service_class,
        fxt_mock_model_service_class,
        fxt_mock_job_service,
        fxt_mock_model_service,
        fxt_mock_binary_repos,
    ):
        """Test cleanup when training fails and model has export_path."""
        fxt_job.payload = {"model_name": "padim"}
        fxt_mock_job_service.get_pending_train_job.return_value = fxt_job
        fxt_model.export_path = "/path/to/model"

        with patch("services.training_service.asyncio.to_thread") as mock_to_thread:
            # Mock the training to succeed first, setting export_path, then fail
            def mock_train_model(cls, model, device=None):
                model.export_path = "/path/to/model"
                raise Exception("Training failed")

            mock_to_thread.side_effect = mock_train_model

            with pytest.raises(Exception, match="Training failed"):
                asyncio.run(TrainingService.train_pending_job())

            # Verify cleanup was called
            fxt_mock_binary_repos[0].return_value.delete_model_folder.assert_called_once()
            fxt_mock_model_service.delete_model.assert_called_once()
            call_args = fxt_mock_model_service.delete_model.call_args
            assert call_args[1]["project_id"] == fxt_job.project_id

    def test_train_model_success(
        self,
        fxt_model,
        fxt_model_binary_repo,
        fxt_image_binary_repo,
        fxt_mock_anomalib_components,
        fxt_mock_binary_repos,
    ):
        """Test successful model training with platform-specific worker configuration."""
        # Setup platform mock
        # Setup binary repo paths
        fxt_image_binary_repo.project_folder_path = "/path/to/images"
        fxt_model_binary_repo.model_folder_path = "/path/to/model"

        # Call the method
        result = TrainingService._train_model(fxt_model)

        # Verify the result
        assert result == fxt_model
        assert fxt_model.is_ready is True
        assert fxt_model.export_path == "/path/to/model"

        # Verify all components were called correctly
        fxt_mock_anomalib_components["folder_class"].assert_called_once()
        fxt_mock_anomalib_components["get_model"].assert_called_once_with(model=fxt_model.name)
        
        # Verify Engine was called with expected parameters
        fxt_mock_anomalib_components["engine_class"].assert_called_once()
        call_args = fxt_mock_anomalib_components["engine_class"].call_args
        assert call_args[1]["default_root_dir"] == "/path/to/model"
        assert "logger" in call_args[1]
        assert len(call_args[1]["logger"]) == 2  # trackio and tensorboard
        assert call_args[1]["max_epochs"] == 10
        
        fxt_mock_anomalib_components["engine"].train.assert_called_once_with(
            model=fxt_mock_anomalib_components["anomalib_model"], datamodule=fxt_mock_anomalib_components["folder"]
        )
        fxt_mock_anomalib_components["engine"].export.assert_called_once()
