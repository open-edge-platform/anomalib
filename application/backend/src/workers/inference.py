# Copyright (C) 2025-2026 Intel Corporation
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
    from collections.abc import Callable
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
        self._model_check_interval: float = 5.0  # seconds between model refresh checks
        self._is_passthrough_mode: bool = False
        self._overlay = True

    def setup(self) -> None:
        super().setup()
        self._metrics_service = MetricsService(self._shm_name, self._shm_lock)

    async def _get_active_model(self) -> LoadedModel | None:
        try:
            async with get_async_db_session_ctx() as session:
                repo = PipelineRepository(session)
                pipeline = await repo.get_active_pipeline()
                # Passthrough mode: pipeline is active but not running, so bypass inference
                self._is_passthrough_mode = pipeline is None or (
                    pipeline.status.is_active and not pipeline.status.is_running
                )
                self._overlay = pipeline.overlay if pipeline and pipeline.overlay is not None else self._overlay
                logger.debug(f"Passthrough mode {'activated' if self._is_passthrough_mode else 'disabled'}.")
                if pipeline is None or pipeline.model is None:
                    return None

                model = pipeline.model
                return LoadedModel(name=model.name, id=model.id, model=model, device=pipeline.inference_device)
        except Exception as e:
            logger.error(f"Failed to query active pipeline/model: {e}", exc_info=True)
            return None

    async def _model_refresh_daemon(self) -> None:
        """Background daemon that periodically refreshes the active model reference.

        Runs independently every _model_check_interval seconds to detect
        configuration changes without impacting frame processing performance.
        """
        while not self.should_stop():
            await asyncio.sleep(self._model_check_interval)

            try:
                previous_model_id = self._loaded_model.id if self._loaded_model else None
                self._loaded_model = await self._get_active_model()

                # Log model transitions
                new_model_id = self._loaded_model.id if self._loaded_model else None
                if previous_model_id != new_model_id:
                    if new_model_id:
                        logger.info(
                            f"Model refresh daemon: Active model changed to "
                            f"'{self._loaded_model.name}' ({new_model_id})",  # type: ignore[union-attr]
                        )
                    else:
                        logger.info("Model refresh daemon: Switched to passthrough mode (no active model)")
                logger.debug(f"Model refresh daemon running in {self._model_check_interval}s")
            except Exception as e:
                logger.error(f"Model refresh daemon error: {e}", exc_info=True)
                # Continue running despite errors

    async def _handle_model_reload(self) -> None:
        # Handle model reload signal: force reload currently active model
        if self._model_reload_event.is_set():
            # The loop handles the case when the active model is switched again while reloading
            while self._model_reload_event.is_set():
                self._model_reload_event.clear()
                logger.info("Model reload event detected; reloading active model")
                self._loaded_model = await self._get_active_model()

                if self._loaded_model is None:
                    raise RuntimeError("No active model configured.")
                # Remove cached model to force reload
                try:
                    self._cached_models.pop(self._loaded_model.id, None)
                except Exception as e:
                    model_id = getattr(self._loaded_model, "id", "unknown")
                    logger.debug(f"Failed to evict cached model {model_id}: {e}")
                # Preload the model for faster first inference
                try:
                    inferencer = await ModelService.load_inference_model(
                        self._loaded_model.model,
                        device=self._loaded_model.device,
                    )
                    self._cached_models[self._loaded_model.id] = inferencer
                    logger.info(
                        f"Reloaded inference model '{self._loaded_model.name}' ({self._loaded_model.id}) "
                        f"on device `{self._loaded_model.device}`",
                    )
                except DeviceNotFoundError:
                    # Load model using the default device
                    logger.warning(
                        f"Device '{self._loaded_model.device}' not found; "
                        f"loading model '{self._loaded_model.name}' ({self._loaded_model.id}) on default device",
                    )
                    inferencer = await ModelService.load_inference_model(self._loaded_model.model)
                    self._cached_models[self._loaded_model.id] = inferencer
                except Exception as e:
                    logger.error(f"Failed to reload model '{self._loaded_model.name}': {e}", exc_info=True)
                    # Leave cache empty; next predict will attempt to load again

    async def _get_next_frame(self) -> StreamData | None:
        """Retrieve the next frame from the queue."""
        try:
            return self._frame_queue.get(timeout=1)
        except std_queue.Empty:
            logger.debug("No frame available yet")
            return None
        except Exception as e:
            logger.error(f"Failed to get frame from queue: {e}")
            return None

    async def _handle_passthrough_mode(self, stream_data: StreamData) -> None:
        """Handle frame in passthrough mode (no model loaded)."""
        try:
            self._pred_queue.put(stream_data, timeout=1)
        except std_queue.Full:
            logger.debug("Prediction queue is full (passthrough mode); dropping frame")

    async def _encode_frame(self, frame: np.ndarray) -> bytes | None:
        """Encode frame to JPEG bytes for inference."""
        try:
            success, buf = cv2.imencode(".jpg", frame)
            if not success:
                logger.warning("Failed to encode frame; skipping")
                return None
            return buf.tobytes()
        except Exception as e:
            logger.error(f"Error encoding frame: {e}", exc_info=True)
            return None

    async def _run_inference(self, image_bytes: bytes) -> Any | None:
        """Run inference on encoded image and record metrics."""
        if self._loaded_model is None:
            logger.error("Cannot run inference: no active model configured")
            return None

        start_t = MetricsService.record_inference_start()
        try:
            return await ModelService.predict_image(
                self._loaded_model.model,
                image_bytes,
                self._cached_models,  # type: ignore[arg-type]
                device=self._loaded_model.device,
            )
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            return None
        finally:
            try:
                if self._metrics_service is not None and self._loaded_model is not None:
                    self._metrics_service.record_inference_end(self._loaded_model.id, start_t)
            except Exception as e:
                logger.debug(f"Failed to record inference metric: {e}")

    async def _process_frame_with_model(self, stream_data: StreamData) -> None:
        """Process frame with loaded model and attach inference data.

        Raises:
            ValueError: If frame data is invalid or encoding fails
            RuntimeError: If inference or model operations fail
        """
        frame = stream_data.frame_data
        if frame is None:
            raise ValueError("Received empty frame data")

        # Encode frame
        image_bytes = await self._encode_frame(frame)
        if image_bytes is None:
            raise ValueError("Failed to encode frame to JPEG")

        # Run inference
        prediction_response = await self._run_inference(image_bytes)
        if prediction_response is None or self._loaded_model is None:
            raise RuntimeError("Inference failed or model became unavailable")

        # Build visualization
        overlays: list[Callable] = []
        if self._overlay:
            overlays.append(Visualizer.overlay_anomaly_heatmap)
        overlays.append(Visualizer.draw_prediction_label)
        vis_frame: np.ndarray = Visualizer.overlay_predictions(frame, prediction_response, *overlays)

        # Package inference data
        stream_data.inference_data = InferenceData(
            prediction=prediction_response,  # type: ignore[arg-type]
            visualized_prediction=vis_frame,
            model_name=self._loaded_model.name,
        )

    async def _enqueue_result(self, stream_data: StreamData) -> None:
        """Enqueue processed result to prediction queue."""
        try:
            self._pred_queue.put(stream_data, timeout=1)
        except std_queue.Full:
            logger.debug("Prediction queue is full; dropping result")

    @logger.catch()
    async def run_loop(self) -> None:
        """Main processing loop for inference worker."""
        # Initialize model reference on startup
        self._loaded_model = await self._get_active_model()

        # Start background daemon for periodic model refresh
        refresh_task = asyncio.create_task(self._model_refresh_daemon())

        try:
            while not self.should_stop():
                # Pull next frame
                stream_data = await self._get_next_frame()
                if stream_data is None:
                    continue

                # Handle model reload (immediate refresh on events)
                try:
                    await self._handle_model_reload()
                except Exception as e:
                    logger.warning(f"Model reload handling failed: {e}")
                    await asyncio.sleep(1)
                    continue

                # Check passthrough mode (no inference)
                if self._is_passthrough_mode:
                    await self._handle_passthrough_mode(stream_data)
                    continue

                if self._loaded_model is None:
                    logger.error("Cannot run inference: model is not loaded: retrying in 1 second")
                    await asyncio.sleep(1)
                    continue

                # Process frame with model and enqueue result
                try:
                    await self._process_frame_with_model(stream_data)
                    await self._enqueue_result(stream_data)
                except (ValueError, RuntimeError) as e:
                    logger.warning(f"Frame processing skipped: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error during frame processing: {e}", exc_info=True)
                    continue
        finally:
            # Clean shutdown: cancel daemon task
            refresh_task.cancel()
            try:
                await refresh_task
            except asyncio.CancelledError:
                logger.info("Model refresh daemon stopped")
