# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from anomalib.deploy import ExportType, OpenVINOInferencer
from PIL import Image

from models import PredictionLabel
from repositories import ModelRepository
from repositories.binary_repo import ModelBinaryRepository
from services import ModelService


@pytest.fixture
def fxt_model_repository():
    """Fixture for a mock model repository."""
    return MagicMock(spec=ModelRepository)


@pytest.fixture
def fxt_model_binary_repo():
    """Fixture for a mock model binary repository."""
    return MagicMock(spec=ModelBinaryRepository)


@pytest.fixture
def fxt_model_service(fxt_db_session, fxt_model_repository, fxt_model_binary_repo):
    """Fixture for ModelService with mocked repositories."""
    service = ModelService(fxt_db_session)
    # Mock the repository method to return our mock
    service.repository = MagicMock(return_value=fxt_model_repository)
    return service


class TestModelService:
    def test_create_model(self, fxt_model_service, fxt_model_repository, fxt_model):
        """Test creating a model."""
        fxt_model_repository.save.return_value = fxt_model

        result = asyncio.run(fxt_model_service.create_model(fxt_model))

        assert result == fxt_model
        fxt_model_repository.save.assert_called_once_with(fxt_model)

    def test_get_model_list(self, fxt_model_service, fxt_model_repository, fxt_model_list, fxt_project):
        """Test getting model list."""
        fxt_model_repository.get_all.return_value = fxt_model_list.models

        result = asyncio.run(fxt_model_service.get_model_list(fxt_project.id))

        assert result == fxt_model_list
        fxt_model_repository.get_all.assert_called_once()

    def test_get_model_by_id(self, fxt_model_service, fxt_model_repository, fxt_model, fxt_project):
        """Test getting model by ID."""
        fxt_model_repository.get_by_id.return_value = fxt_model

        result = asyncio.run(fxt_model_service.get_model_by_id(fxt_project.id, fxt_model.id))

        assert result == fxt_model
        fxt_model_repository.get_by_id.assert_called_once_with(fxt_model.id)

    def test_get_model_by_id_not_found(self, fxt_model_service, fxt_model_repository, fxt_project):
        """Test getting model by ID when not found."""
        fxt_model_repository.get_by_id.return_value = None

        result = asyncio.run(fxt_model_service.get_model_by_id(fxt_project.id, "non-existent-id"))

        assert result is None
        fxt_model_repository.get_by_id.assert_called_once_with("non-existent-id")

    def test_delete_model(self, fxt_model_service, fxt_model_repository, fxt_model, fxt_project):
        """Test deleting a model."""
        fxt_model_repository.delete_by_id.return_value = None

        asyncio.run(fxt_model_service.delete_model(fxt_project.id, fxt_model.id))

        fxt_model_repository.delete_by_id.assert_called_once_with(fxt_model.id)

    def test_load_inference_model_success(self, fxt_model_service, fxt_model, fxt_openvino_inferencer):
        """Test loading inference model successfully."""
        with patch("services.model_service.ModelBinaryRepository") as mock_bin_repo_class:
            mock_bin_repo = MagicMock()
            mock_bin_repo_class.return_value = mock_bin_repo
            mock_bin_repo.get_weights_file_path.return_value = "/path/to/model.xml"

            with patch("services.model_service.asyncio.to_thread") as mock_to_thread:
                mock_to_thread.return_value = fxt_openvino_inferencer

                result = asyncio.run(fxt_model_service.load_inference_model(fxt_model, "CPU"))

                assert result == fxt_openvino_inferencer
                mock_bin_repo.get_weights_file_path.assert_called_once_with(
                    format=ExportType.OPENVINO, name="model.xml"
                )
                mock_to_thread.assert_called_once_with(OpenVINOInferencer, path="/path/to/model.xml", device="CPU")

    def test_load_inference_model_unsupported_format(self, fxt_model_service, fxt_model):
        """Test loading inference model with unsupported format."""
        fxt_model.format = "unsupported_format"

        with pytest.raises(NotImplementedError) as exc_info:
            asyncio.run(fxt_model_service.load_inference_model(fxt_model))

        assert "Model format unsupported_format is not supported" in str(exc_info.value)

    def test_predict_image_with_cached_model(
        self, fxt_model_service, fxt_model, fxt_image_bytes, fxt_prediction_response, fxt_openvino_inferencer
    ):
        """Test prediction with cached model."""
        cached_models = {fxt_model.id: fxt_openvino_inferencer}

        with patch.object(fxt_model_service, "_run_prediction_pipeline") as mock_pipeline:
            mock_pipeline.return_value = {
                "anomaly_map": "base64_encoded_image",
                "label": PredictionLabel.NORMAL,
                "score": 0.1,
            }

            result = asyncio.run(fxt_model_service.predict_image(fxt_model, fxt_image_bytes, cached_models))

            assert result.anomaly_map == "base64_encoded_image"
            assert result.label == PredictionLabel.NORMAL
            assert result.score == 0.1
            mock_pipeline.assert_called_once()

    def test_predict_image_without_cached_model(
        self, fxt_model_service, fxt_model, fxt_image_bytes, fxt_openvino_inferencer
    ):
        """Test prediction without cached model."""
        with patch.object(fxt_model_service, "load_inference_model") as mock_load_model:
            mock_load_model.return_value = fxt_openvino_inferencer

            with patch.object(fxt_model_service, "_run_prediction_pipeline") as mock_pipeline:
                mock_pipeline.return_value = {
                    "anomaly_map": "base64_encoded_image",
                    "label": PredictionLabel.NORMAL,
                    "score": 0.1,
                }

                result = asyncio.run(fxt_model_service.predict_image(fxt_model, fxt_image_bytes, {}))

                assert result.anomaly_map == "base64_encoded_image"
                assert result.label == PredictionLabel.NORMAL
                assert result.score == 0.1
                mock_load_model.assert_called_once_with(fxt_model)
                mock_pipeline.assert_called_once()

    def test_predict_image_caches_model(self, fxt_model_service, fxt_model, fxt_image_bytes, fxt_openvino_inferencer):
        """Test that prediction caches the loaded model."""
        cached_models = {}

        with patch.object(fxt_model_service, "load_inference_model") as mock_load_model:
            mock_load_model.return_value = fxt_openvino_inferencer

            with patch.object(fxt_model_service, "_run_prediction_pipeline") as mock_pipeline:
                mock_pipeline.return_value = {
                    "anomaly_map": "base64_encoded_image",
                    "label": PredictionLabel.NORMAL,
                    "score": 0.1,
                }

                asyncio.run(fxt_model_service.predict_image(fxt_model, fxt_image_bytes, cached_models))

                assert fxt_model.id in cached_models
                assert cached_models[fxt_model.id] == fxt_openvino_inferencer

    @patch("services.model_service.cv2.imdecode")
    @patch("services.model_service.cv2.cvtColor")
    def test_run_prediction_pipeline(self, mock_cvt_color, mock_imdecode, fxt_openvino_inferencer, fxt_image_bytes):
        """Test the prediction pipeline static method."""
        # Mock OpenCV functions to avoid actual image processing
        mock_imdecode.return_value = np.array(
            [[[100, 100, 100], [200, 200, 200]], [[150, 150, 150], [250, 250, 250]]], dtype=np.uint8
        )
        mock_cvt_color.return_value = np.array(
            [[[100, 100, 100], [200, 200, 200]], [[150, 150, 150], [250, 250, 250]]], dtype=np.uint8
        )

        # Create a test anomaly map
        test_anomaly_map = np.array([[[0.1, 0.2], [0.3, 0.4]]])
        fxt_openvino_inferencer.predict.return_value.anomaly_map = test_anomaly_map
        fxt_openvino_inferencer.predict.return_value.pred_label.item.return_value = 0
        fxt_openvino_inferencer.predict.return_value.pred_score.item.return_value = 0.1

        result = ModelService._run_prediction_pipeline(fxt_openvino_inferencer, fxt_image_bytes)

        assert "anomaly_map" in result
        assert result["label"] == PredictionLabel.NORMAL
        assert result["score"] == 0.1
        assert isinstance(result["anomaly_map"], str)  # Should be base64 encoded

    @patch("services.model_service.cv2.imdecode")
    @patch("services.model_service.cv2.cvtColor")
    def test_run_prediction_pipeline_anomalous(
        self, mock_cvt_color, mock_imdecode, fxt_openvino_inferencer, fxt_image_bytes
    ):
        """Test the prediction pipeline with anomalous prediction."""
        # Mock OpenCV functions to avoid actual image processing
        mock_imdecode.return_value = np.array(
            [[[100, 100, 100], [200, 200, 200]], [[150, 150, 150], [250, 250, 250]]], dtype=np.uint8
        )
        mock_cvt_color.return_value = np.array(
            [[[100, 100, 100], [200, 200, 200]], [[150, 150, 150], [250, 250, 250]]], dtype=np.uint8
        )

        # Create a test anomaly map
        test_anomaly_map = np.array([[[0.8, 0.9], [0.7, 0.6]]])
        fxt_openvino_inferencer.predict.return_value.anomaly_map = test_anomaly_map
        fxt_openvino_inferencer.predict.return_value.pred_label.item.return_value = 1
        fxt_openvino_inferencer.predict.return_value.pred_score.item.return_value = 0.9

        result = ModelService._run_prediction_pipeline(fxt_openvino_inferencer, fxt_image_bytes)

        assert "anomaly_map" in result
        assert result["label"] == PredictionLabel.ANOMALOUS
        assert result["score"] == 0.9
        assert isinstance(result["anomaly_map"], str)  # Should be base64 encoded

    @patch("services.model_service.cv2.imdecode")
    @patch("services.model_service.cv2.cvtColor")
    def test_run_prediction_pipeline_image_processing(self, mock_cvt_color, mock_imdecode, fxt_openvino_inferencer):
        """Test that image processing works correctly in the pipeline."""
        # Test with actual image bytes
        test_image = Image.new("RGB", (100, 100), color="red")
        with io.BytesIO() as buf:
            test_image.save(buf, format="JPEG")
            image_bytes = buf.getvalue()

        # Mock OpenCV functions to avoid actual image processing
        mock_imdecode.return_value = np.array(
            [[[100, 100, 100], [200, 200, 200]], [[150, 150, 150], [250, 250, 250]]], dtype=np.uint8
        )
        mock_cvt_color.return_value = np.array(
            [[[100, 100, 100], [200, 200, 200]], [[150, 150, 150], [250, 250, 250]]], dtype=np.uint8
        )

        # Create a test anomaly map
        test_anomaly_map = np.array([[[0.1, 0.2], [0.3, 0.4]]])
        fxt_openvino_inferencer.predict.return_value.anomaly_map = test_anomaly_map
        fxt_openvino_inferencer.predict.return_value.pred_label.item.return_value = 0
        fxt_openvino_inferencer.predict.return_value.pred_score.item.return_value = 0.1

        result = ModelService._run_prediction_pipeline(fxt_openvino_inferencer, image_bytes)

        # Verify that the inferencer was called with a numpy array
        fxt_openvino_inferencer.predict.assert_called_once()
        call_args = fxt_openvino_inferencer.predict.call_args[0][0]
        assert isinstance(call_args, np.ndarray)
        assert call_args.shape[2] == 3  # RGB image

        # Verify the result structure
        assert "anomaly_map" in result
        assert result["label"] == PredictionLabel.NORMAL
        assert result["score"] == 0.1
