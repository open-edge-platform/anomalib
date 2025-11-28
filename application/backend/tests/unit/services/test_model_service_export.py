from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from anomalib.deploy import CompressionType, ExportType

from pydantic_models import Model
from pydantic_models.model import ExportParameters
from services import ResourceNotFoundError
from services.model_service import ModelService


@pytest.fixture
def mock_model():
    return Model(
        id=uuid4(),
        project_id=uuid4(),
        name="Padim",
        format=ExportType.OPENVINO,
        is_ready=True,
        train_job_id=uuid4(),
        dataset_snapshot_id=uuid4(),
    )


@pytest.fixture
def mock_model_service():
    return ModelService()


@pytest.mark.asyncio
@patch("services.model_service.ModelService.get_model_by_id")
@patch("services.model_service.ModelBinaryRepository")
@patch("services.model_service.Engine")
@patch("services.model_service.get_model")
@patch("services.model_service.shutil.make_archive")
@patch("services.model_service.asyncio.to_thread")
async def test_export_model_success(
    mock_to_thread,
    mock_make_archive,
    mock_get_model,
    mock_engine_cls,
    mock_binary_repo_cls,
    mock_get_model_by_id,
    mock_model_service,
    mock_model,
    tmp_path,
):
    """Test successful model export."""
    # Setup mocks
    mock_get_model_by_id.return_value = mock_model

    # Setup binary repo mock
    mock_binary_repo = MagicMock()
    mock_binary_repo.model_folder_path = str(tmp_path)
    mock_binary_repo_cls.return_value = mock_binary_repo

    # Create dummy checkpoint file
    ckpt_dir = tmp_path / "weights" / "lightning"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "model.ckpt").touch()

    # Setup export params
    export_params = ExportParameters(format=ExportType.OPENVINO, compression=CompressionType.FP16)

    # Setup to_thread result to match expected return
    expected_zip_path = (
        Path("data/exports")
        / str(mock_model.id)
        / mock_model.name.title()
        / f"{mock_model.project_id}-{mock_model.name}"
        / f"{mock_model.name}_{export_params.format.value}_{export_params.compression.value}.zip"
    )
    mock_to_thread.return_value = expected_zip_path

    # Call method
    result = await mock_model_service.export_model(mock_model.project_id, mock_model.id, export_params)

    # Verify results
    assert result == expected_zip_path
    mock_to_thread.assert_called_once()


@pytest.mark.asyncio
@patch("services.model_service.ModelService.get_model_by_id")
async def test_export_model_not_found(mock_get_model_by_id, mock_model_service):
    """Test export when model does not exist."""
    mock_get_model_by_id.return_value = None

    export_params = ExportParameters(format=ExportType.OPENVINO)

    with pytest.raises(ResourceNotFoundError):
        await mock_model_service.export_model(uuid4(), uuid4(), export_params)


@pytest.mark.asyncio
@patch("services.model_service.ModelService.get_model_by_id")
@patch("services.model_service.ModelBinaryRepository")
async def test_export_model_ckpt_not_found(
    mock_binary_repo_cls, mock_get_model_by_id, mock_model_service, mock_model, tmp_path
):
    """Test export when checkpoint file is missing."""
    mock_get_model_by_id.return_value = mock_model

    mock_binary_repo = MagicMock()
    mock_binary_repo.model_folder_path = str(tmp_path)
    mock_binary_repo_cls.return_value = mock_binary_repo

    # Checkpoint file intentionally omitted to test error handling

    export_params = ExportParameters(format=ExportType.OPENVINO)

    with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
        await mock_model_service.export_model(mock_model.project_id, mock_model.id, export_params)


@pytest.mark.asyncio
@patch("services.model_service.ModelService.get_model_by_id")
async def test_export_model_cached_logic(mock_get_model_by_id, mock_model_service, mock_model):
    """Test that cached export is returned if it exists."""
    mock_get_model_by_id.return_value = mock_model

    export_params = ExportParameters(format=ExportType.OPENVINO)

    with patch("services.model_service.Path") as mock_path_cls:
        # Setup the mock path chain
        mock_path_instance = MagicMock()
        mock_path_cls.return_value = mock_path_instance

        # Mock / operators
        mock_path_instance.__truediv__.return_value = mock_path_instance

        # Verify existence check returns True
        mock_path_instance.exists.return_value = True

        result = await mock_model_service.export_model(mock_model.project_id, mock_model.id, export_params)

        # Should return the path immediately without calling run_export
        assert result == mock_path_instance
