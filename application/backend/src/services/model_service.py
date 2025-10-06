# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import io
import logging
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventClass
from uuid import UUID

import cv2
import numpy as np
from anomalib.deploy import ExportType, OpenVINOInferencer
from PIL import Image

from db import get_async_db_session_ctx
from pydantic_models import Model, ModelList, PredictionLabel, PredictionResponse
from repositories import ModelRepository
from repositories.binary_repo import ModelBinaryRepository

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    name: str
    id: UUID
    model: Model


class ModelService:
    """
    Service for managing models and inference operations.

    Handles model CRUD operations, loading inference models, and running
    predictions on images. Uses asyncio.to_thread for CPU-intensive operations
    to maintain event loop responsiveness.
    """

    def __init__(self, mp_model_reload_event: EventClass | None = None) -> None:
        self._mp_model_reload_event = mp_model_reload_event

    def activate_model(self) -> None:
        """Notify workers to (re)load the active model.

        Sets the shared multiprocessing event so the inference worker reloads
        the current active model lazily and only once.
        """
        try:
            if self._mp_model_reload_event is not None:
                self._mp_model_reload_event.set()
        except Exception as e:
            # Best-effort signaling; avoid bubbling up to API layer
            logger.debug("Failed to signal model reload event: %s", e)

    @staticmethod
    async def create_model(model: Model) -> Model:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session, project_id=model.project_id)
            return await repo.save(model)

    @staticmethod
    async def get_model_list(project_id: UUID) -> ModelList:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session, project_id=project_id)
            return ModelList(models=await repo.get_all())

    @staticmethod
    async def get_model_by_id(project_id: UUID, model_id: UUID) -> Model | None:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session, project_id=project_id)
            return await repo.get_by_id(model_id)

    @staticmethod
    async def delete_model(project_id: UUID, model_id: UUID) -> None:
        async with get_async_db_session_ctx() as session:
            repo = ModelRepository(session, project_id=project_id)
            return await repo.delete_by_id(model_id)

    @staticmethod
    async def load_inference_model(model: Model, device: str = "CPU") -> OpenVINOInferencer:
        """Load a model for inference using the anomalib OpenVINO inferencer."""
        if model.format is not ExportType.OPENVINO:
            raise NotImplementedError(f"Model format {model.format} is not supported for inference at this moment.")

        model_bin_repo = ModelBinaryRepository(project_id=model.project_id, model_id=model.id)
        model_path = model_bin_repo.get_weights_file_path(format=model.format, name="model.xml")
        return await asyncio.to_thread(OpenVINOInferencer, path=model_path, device=device)

    async def predict_image(
        self, model: Model, image_bytes: bytes, cached_models: dict[UUID, OpenVINOInferencer] | None = None
    ) -> PredictionResponse:
        """
        Run prediction on an image using the specified model.

        Uses asyncio.to_thread to run the entire CPU-intensive prediction pipeline
        in a single thread, maintaining event loop responsiveness.

        Args:
            model: The model to use for prediction
            image_bytes: Raw image bytes from uploaded file
            cached_models: Optional dict to cache loaded models (for performance)

        Returns:
            PredictionResponse: Structured prediction results
        """
        # Use cached model if available, otherwise load it
        if cached_models and model.id in cached_models:
            inference_model = cached_models[model.id]
        else:
            inference_model = await self.load_inference_model(model)
            if cached_models is not None:
                cached_models[model.id] = inference_model

        # Run entire prediction pipeline in a single thread
        # This includes image processing, model inference, and result processing
        response_data = await asyncio.to_thread(self._run_prediction_pipeline, inference_model, image_bytes)

        return PredictionResponse(**response_data)

    @staticmethod
    def _run_prediction_pipeline(inference_model: OpenVINOInferencer, image_bytes: bytes) -> dict:
        """Run the complete prediction pipeline in a single thread."""
        # Process image
        npd = np.frombuffer(image_bytes, np.uint8)
        bgr_image = cv2.imdecode(npd, -1)
        numpy_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Run prediction
        pred = inference_model.predict(numpy_image)

        # Process anomaly map
        arr = pred.anomaly_map.squeeze()  # Remove dimensions of size 1
        arr_normalized = (arr * 255).astype(np.uint8)  # Normalize to 0-255 and convert to uint8
        im = Image.fromarray(arr_normalized)  # Automatically detects grayscale mode

        # Convert to base64
        with io.BytesIO() as buf:
            im.save(buf, format="PNG")
            im_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Create response data
        label = PredictionLabel.ANOMALOUS if pred.pred_label.item() else PredictionLabel.NORMAL
        score = float(pred.pred_score.item())

        return {"anomaly_map": im_base64, "label": label, "score": score}
