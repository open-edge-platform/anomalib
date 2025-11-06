# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import queue as std_queue
from typing import TYPE_CHECKING, Any

import cv2
import loguru
from loguru import logger

if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.synchronize import Event as EventClass
    from multiprocessing.synchronize import Lock

    import numpy as np


from db import get_async_db_session_ctx
from entities.stream_data import InferenceData, StreamData
from repositories import PipelineRepository
from services import ModelService
from services.exceptions import DeviceNotFoundError
from services.metrics_service import MetricsService
from services.model_service import LoadedModel
from utils.visualization import Visualizer
from workers.base import BaseProcessWorker


class InferenceWorker(BaseProcessWorker):
    """A process that pulls frames from the frame queue, runs inference, and pushes results to the prediction queue."""

    ROLE = "Inference"

    def __init__(
        self,
        frame_queue: mp.Queue,
        pred_queue: mp.Queue,
        stop_event: EventClass,
        model_reload_event: EventClass,
        shm_name: str,
        shm_lock: Lock,
        logger_: loguru.Logger | None = None,
    ) -> None:
        super().__init__(stop_event=stop_event, queues_to_cancel=[pred_queue], logger_=logger_)
        self._frame_queue = frame_queue
        self._pred_queue = pred_queue
        self._model_reload_event = model_reload_event
        self._shm_name = shm_name
        self._shm_lock = shm_lock

        self._metrics_service: MetricsService | None = None
        self._loaded_model: LoadedModel | None = None
        self._last_model_obj_id = 0  # track the id of the Model object to install the callback only once
        self._cached_models: dict[Any, object] = {}

    def setup(self) -> None:
        super().setup()
        self._metrics_service = MetricsService(self._shm_name, self._shm_lock)

    @staticmethod
    async def _get_active_model() -> LoadedModel | None:
        try:
            async with get_async_db_session_ctx() as session:
                repo = PipelineRepository(session)
                pipeline = await repo.get_active_pipeline()
                if pipeline is None or pipeline.model is None:
                    return None
                model = pipeline.model
                return LoadedModel(name=model.name, id=model.id, model=model, device=pipeline.inference_device)
        except Exception as e:
            logger.error(f"Failed to query active pipeline/model: {e}", exc_info=True)
            return None

    async def _handle_model_reload(self) -> None:
        # Handle model reload signal: force reload currently active model
        try:
            if self._model_reload_event.is_set():
                # The loop handles the case when the active model is switched again while reloading
                while self._model_reload_event.is_set():
                    self._model_reload_event.clear()

                    if self._loaded_model is None:
                        continue

                    # Remove cached model to force reload
                    try:
                        self._cached_models.pop(self._loaded_model.id, None)
                    except Exception as e:
                        logger.debug(
                            "Failed to evict cached model %s: %s",
                            getattr(self._loaded_model, "id", "unknown"),
                            e,
                        )
                    # Preload the model for faster first inference
                    try:
                        inferencer = await ModelService.load_inference_model(
                            self._loaded_model.model, device=self._loaded_model.device
                        )
                        self._cached_models[self._loaded_model.id] = inferencer
                        logger.info(
                            "Reloaded inference model '%s' (%s) on device %s",
                            self._loaded_model.name,
                            self._loaded_model.id,
                            self._loaded_model.device,
                        )
                    except DeviceNotFoundError:
                        # Load model using the default device
                        logger.warning(
                            "Device '%s' not found; loading model '%s' (%s) on default device",
                            self._loaded_model.device,
                            self._loaded_model.name,
                            self._loaded_model.id,
                        )
                        inferencer = await ModelService.load_inference_model(self._loaded_model.model)
                        self._cached_models[self._loaded_model.id] = inferencer
                    except Exception as e:
                        logger.error(f"Failed to reload model '{self._loaded_model.name}': {e}", exc_info=True)
                        # Leave cache empty; next predict will attempt to load again
        except Exception as e:
            logger.debug(f"Error while handling model reload event: {e}")

    @logger.catch()
    async def run_loop(self) -> None:  # noqa: PLR0912, PLR0915
        while not self.should_stop():
            # Ensure model is loaded/selected from active pipeline
            active_model = await self._get_active_model()
            if active_model is None:
                logger.trace("No active model configured; retrying in 1 second")
                await asyncio.sleep(1)
                continue

            # Refresh loaded model reference if changed
            if self._loaded_model is None or self._loaded_model.id != active_model.id:
                self._loaded_model = active_model
                logger.info(f"Using model '{self._loaded_model.name}' ({self._loaded_model.id}) for inference")

            await self._handle_model_reload()

            # Pull next frame
            try:
                stream_data: StreamData = self._frame_queue.get(timeout=1)
            except std_queue.Empty:
                logger.debug("No frame available for inference yet")
                continue
            except Exception:
                logger.debug("No frame available for inference yet")
                continue

            # Prepare input bytes for ModelService.predict_image (expects encoded image bytes)
            frame = stream_data.frame_data
            if frame is None:
                logger.debug("Received empty frame; skipping")
                continue

            try:
                success, buf = cv2.imencode(".jpg", frame)
                if not success:
                    logger.warning("Failed to encode frame; skipping")
                    continue
                image_bytes = buf.tobytes()
            except Exception as e:
                logger.error(f"Error encoding frame: {e}", exc_info=True)
                continue

            # Run inference and collect latency metric
            start_t = MetricsService.record_inference_start()
            try:
                prediction_response = await ModelService.predict_image(
                    self._loaded_model.model,
                    image_bytes,
                    self._cached_models,  # type: ignore[arg-type]
                )
            except Exception as e:
                logger.error(f"Inference failed: {e}", exc_info=True)
                continue
            finally:
                try:
                    if self._metrics_service is not None and self._loaded_model is not None:
                        self._metrics_service.record_inference_end(self._loaded_model.id, start_t)
                except Exception as e:
                    logger.debug(f"Failed to record inference metric: {e}")

            # Build visualization via utility (no direct overlay/manipulation here)
            vis_frame: np.ndarray = Visualizer.overlay_predictions(frame, prediction_response)

            # Package inference data into stream payload
            try:
                stream_data.inference_data = InferenceData(
                    prediction=prediction_response,  # type: ignore[arg-type]
                    visualized_prediction=vis_frame,
                    model_name=self._loaded_model.name,
                )
            except Exception as e:
                logger.error(f"Failed to attach inference data: {e}", exc_info=True)
                continue

            # Enqueue for downstream dispatchers/visualization
            try:
                self._pred_queue.put(stream_data, timeout=1)
            except std_queue.Full:
                logger.debug("Prediction queue is full; dropping result")
                continue
