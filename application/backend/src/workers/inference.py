# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
import multiprocessing as mp
import queue as std_queue
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.synchronize import Lock
from typing import TYPE_CHECKING, Any

import cv2

from workers.base import BaseProcessWorker

if TYPE_CHECKING:
    import numpy as np

from db import get_async_db_session_ctx
from entities.stream_data import InferenceData, StreamData
from repositories import PipelineRepository
from services import ModelService
from services.metrics_service import MetricsService
from services.model_service import LoadedModel
from utils import log_threads, suppress_child_shutdown_signals
from utils.visualization import Visualizer

import logging
import multiprocessing as mp
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.synchronize import Lock
from typing import Any

logger = logging.getLogger(__name__)


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
    ) -> None:
        super().__init__(stop_event=stop_event, queues_to_cancel=[pred_queue])
        self._frame_queue = frame_queue
        self._pred_queue = pred_queue
        self._model_reload_event = model_reload_event
        self._shm_name = shm_name
        self._shm_lock = shm_lock

        self._metrics_service: MetricsService | None = None
        self._model_service: ModelService | None = None
        self._loaded_model: LoadedModel | None = None
        self._last_model_obj_id = 0  # track the id of the Model object to install the callback only once
        self._cached_models: dict[Any, object] = {}

    def setup(self) -> None:
        self._metrics_service = MetricsService(self._shm_name, self._shm_lock)
        self._model_service = ModelService()

    def _refresh_loaded_model(self) -> LoadedModel | None:
        """
        Get (or reload) the active model. If reloads are requested repeatedly,
        clear the event until the latest model is loaded.
        """
        # If no reload requested, return current model
        if not self._model_reload_event.is_set():
            return self._model_service.get_loaded_inference_model()  # type: ignore

        # Process reload requests - keep reloading until event stabilizes
        loaded_model = None
        while self._model_reload_event.is_set():
            self._model_reload_event.clear()
            loaded_model = self._model_service.get_loaded_inference_model(force_reload=True)  # type: ignore
        return loaded_model

    @staticmethod
    async def _get_active_model() -> LoadedModel | None:
        try:
            async with get_async_db_session_ctx() as session:
                repo = PipelineRepository(session)
                pipeline = await repo.get_active_pipeline()
                if pipeline is None or pipeline.model is None:
                    return None
                model = pipeline.model
                return LoadedModel(name=model.name, id=model.id, model=model)
        except Exception as e:
            logger.error("Failed to query active pipeline/model: %s", e, exc_info=True)
            return None

    async def _handle_model_reload(self) -> None:
        # Handle model reload signal: force reload currently active model
        try:
            if self._model_reload_event.is_set():
                # The loop handles the case when the active model is switched again while reloading
                while self._model_reload_event.is_set():
                    self._model_reload_event.clear()
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
                        inferencer = await self._model_service.load_inference_model(self._loaded_model.model)
                        self._cached_models[self._loaded_model.id] = inferencer
                        logger.info(
                            "Reloaded inference model '%s' (%s)", self._loaded_model.name, self._loaded_model.id
                        )
                    except Exception as e:
                        logger.error("Failed to reload model '%s': %s", self._loaded_model.name, e, exc_info=True)
                        # Leave cache empty; next predict will attempt to load again
        except Exception as e:
            logger.debug("Error while handling model reload event: %s", e)

    async def run_loop(self) -> None:
        try:
            while not self.should_stop():
                # Ensure model is loaded/selected from active pipeline
                active_model = await self._get_active_model()
                if active_model is None:
                    logger.debug("No active model configured; retrying in 1 second")
                    await asyncio.sleep(1)
                    continue

                # Refresh loaded model reference if changed
                if self._loaded_model is None or self._loaded_model.id != active_model.id:
                    self._loaded_model = active_model
                    logger.info(
                        "Using model '%s' (%s) for inference", self._loaded_model.name, self._loaded_model.id
                    )

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
                    logger.error("Error encoding frame: %s", e, exc_info=True)
                    continue

                # Run inference and collect latency metric
                start_t = MetricsService.record_inference_start()
                try:
                    prediction_response = await self._model_service.predict_image(
                        self._loaded_model.model, image_bytes, self._cached_models
                    )
                except Exception as e:
                    logger.error("Inference failed: %s", e, exc_info=True)
                    continue
                finally:
                    try:
                        self._metrics_collector.record_inference_end(self._loaded_model.id, start_t)
                    except Exception as e:
                        logger.debug("Failed to record inference metric: %s", e)

                # Build visualization via utility (no direct overlay/manipulation here)
                vis_frame: np.ndarray = Visualizer.overlay_predictions(frame, prediction_response)

                # Package inference data into stream payload
                try:
                    stream_data.inference_data = InferenceData(
                        prediction=prediction_response,  # type: ignore[assignment]
                        visualized_prediction=vis_frame,
                        model_name=self._loaded_model.name,
                    )
                except Exception as e:
                    logger.error("Failed to attach inference data: %s", e, exc_info=True)
                    continue

                # Enqueue for downstream dispatchers/visualization
                try:
                    self._pred_queue.put(stream_data, timeout=1)
                except std_queue.Full:
                    logger.debug("Prediction queue is full; dropping result")
                    continue
        finally:
            # https://docs.python.org/3/library/multiprocessing.html#all-start-methods
            # section: Joining processes that use queues
            # Call cancel_join_thread() to prevent the parent process from blocking
            # indefinitely when joining child processes that used this queue. This avoids potential
            # deadlocks if the queue's background thread adds more items during the flush.
            if self._pred_queue is not None:
                logger.debug("Cancelling the pred_queue join thread to allow inference process to exit")
                self._pred_queue.cancel_join_thread()

            log_threads(log_level=logging.DEBUG)
            logger.info("Stopped inference routine")
