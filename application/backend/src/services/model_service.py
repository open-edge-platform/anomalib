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
import openvino as ov
import openvino.properties.hint as ov_hints
from anomalib.deploy import ExportType, OpenVINOInferencer
from cachetools.func import lru_cache
from PIL import Image

from db import get_async_db_session_ctx
from pydantic_models import Model, ModelList, PredictionLabel, PredictionResponse
from pydantic_models.model import SupportedDevices
from repositories import ModelRepository
from repositories.binary_repo import ModelBinaryRepository
from services.exceptions import DeviceNotFoundError

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "AUTO"


@dataclass
class LoadedModel:
    name: str
    id: UUID
    model: Model
    device: str | None = None

    def __post_init__(self):
        self.device = self.device or DEFAULT_DEVICE


class ModelService:
    """Service for managing models and inference operations.

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

    @classmethod
    async def load_inference_model(cls, model: Model, device: str | None = None) -> OpenVINOInferencer:
        """Load a model for inference using the anomalib OpenVINO inferencer.

        Args:
            model: The model to load
            device: Device to use for inference. If None, defaults to "AUTO"
        """
        if model.format is not ExportType.OPENVINO:
            raise NotImplementedError(f"Model format {model.format} is not supported for inference at this moment.")

        model_bin_repo = ModelBinaryRepository(project_id=model.project_id, model_id=model.id)
        model_path = model_bin_repo.get_weights_file_path(format=model.format, name="model.xml")
        _device = device or DEFAULT_DEVICE
        try:
            return await asyncio.to_thread(
                OpenVINOInferencer,
                path=model_path,
                device=_device,
                config={ov_hints.performance_mode: ov_hints.PerformanceMode.LATENCY},
            )
        except Exception as e:
            if _device not in cls.get_supported_devices().devices:
                raise DeviceNotFoundError(device_name=_device) from e
            raise e

    async def predict_image(
        self,
        model: Model,
        image_bytes: bytes,
        cached_models: dict[UUID, OpenVINOInferencer] | None = None,
        device: str | None = None,
    ) -> PredictionResponse:
        """Run prediction on an image using the specified model.

        Uses asyncio.to_thread to run the entire CPU-intensive prediction pipeline
        in a single thread, maintaining event loop responsiveness.

        Args:
            model: The model to use for prediction
            image_bytes: Raw image bytes from uploaded file
            cached_models: Optional dict to cache loaded models (for performance)
            device: Optional string indicating the device to use for inference

        Returns:
            PredictionResponse: Structured prediction results
        """
        # Determine if we can use cached model (must exist and device must match if specified)
        use_cached = (
            cached_models is not None
            and model.id in cached_models
            and (device is None or cached_models[model.id].device == device)
        )

        if use_cached:
            inference_model = cached_models[model.id]
        else:
            logger.info(f"Loading model with device: {device or DEFAULT_DEVICE}")
            inference_model = await self.load_inference_model(model, device=device)
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
        arr_scaled = (arr * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
        # convert to color map
        heatmap = cv2.applyColorMap(arr_scaled, cv2.COLORMAP_JET)
        # Add alpha channel with opacity weighted according to the anomaly score
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGBA)
        heatmap[:, :, 3] = arr_scaled

        im = Image.fromarray(heatmap)  # Automatically detects RGBA mode

        # Convert to base64
        with io.BytesIO() as buf:
            im.save(buf, format="PNG")
            im_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Create response data
        label = PredictionLabel.ANOMALOUS if pred.pred_label.item() else PredictionLabel.NORMAL
        score = float(pred.pred_score.item())

        return {"anomaly_map": im_base64, "label": label, "score": score}

    @staticmethod
    @lru_cache
    def get_supported_devices() -> SupportedDevices:
        """Get list of supported devices for inference."""
        core = ov.Core()
        return SupportedDevices(devices=core.available_devices)
