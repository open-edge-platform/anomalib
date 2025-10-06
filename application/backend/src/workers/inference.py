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

logger = logging.getLogger(__name__)


async def _inference_loop(  # noqa: C901, PLR0912, PLR0915
    frame_queue: mp.Queue,
    pred_queue: mp.Queue,
    stop_event: EventClass,
    model_reload_event: EventClass,
    shm_name: str,
    shm_lock: Lock,
) -> None:
    metrics_collector = MetricsService(shm_name, shm_lock)

    model_service = ModelService()
    loaded_model: LoadedModel | None = None
    cached_models: dict[Any, object] = {}

    async def _get_active_model() -> LoadedModel | None:
        async with get_async_db_session_ctx() as session:
            repo = PipelineRepository(session)
            pipeline = await repo.get_active_pipeline()
            if pipeline is None or pipeline.model is None:
                return None
            model = pipeline.model
            return LoadedModel(name=model.name, id=model.id, model=model)

    try:
        while not stop_event.is_set():
            # Exit if parent process died
            parent_process = mp.parent_process()
            if parent_process is not None and not parent_process.is_alive():
                break
            # Ensure model is loaded/selected from active pipeline
            try:
                active_model = await _get_active_model()
            except Exception as e:
                logger.error("Failed to query active pipeline/model: %s", e, exc_info=True)
                active_model = None

            if active_model is None:
                logger.debug("No active model configured; retrying in 1 second")
                await asyncio.sleep(1)
                continue

            # Refresh loaded model reference if changed
            if loaded_model is None or loaded_model.id != active_model.id:
                loaded_model = active_model
                logger.info("Using model '%s' (%s) for inference", loaded_model.name, loaded_model.id)

            # Handle model reload signal: force reload currently active model
            try:
                if model_reload_event.is_set():
                    # The loop handles the case when the active model is switched again while reloading
                    while model_reload_event.is_set():
                        model_reload_event.clear()
                        # Remove cached model to force reload
                        try:
                            cached_models.pop(loaded_model.id, None)
                        except Exception as e:
                            logger.debug(
                                "Failed to evict cached model %s: %s",
                                getattr(loaded_model, "id", "unknown"),
                                e,
                            )
                        # Preload the model for faster first inference
                        try:
                            inferencer = await model_service.load_inference_model(loaded_model.model)
                            cached_models[loaded_model.id] = inferencer
                            logger.info("Reloaded inference model '%s' (%s)", loaded_model.name, loaded_model.id)
                        except Exception as e:
                            logger.error("Failed to reload model '%s': %s", loaded_model.name, e, exc_info=True)
                            # Leave cache empty; next predict will attempt to load again
            except Exception as e:
                logger.debug("Error while handling model reload event: %s", e)

            # Pull next frame
            try:
                stream_data: StreamData = frame_queue.get(timeout=1)
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
                prediction_response = await model_service.predict_image(loaded_model.model, image_bytes, cached_models)
            except Exception as e:
                logger.error("Inference failed: %s", e, exc_info=True)
                continue
            finally:
                try:
                    metrics_collector.record_inference_end(loaded_model.id, start_t)
                except Exception as e:
                    logger.debug("Failed to record inference metric: %s", e)

            # Build visualization via utility (no direct overlay/manipulation here)
            vis_frame: np.ndarray = Visualizer.overlay_predictions(frame, prediction_response)

            # Package inference data into stream payload
            try:
                stream_data.inference_data = InferenceData(
                    prediction=prediction_response,  # type: ignore[assignment]
                    visualized_prediction=vis_frame,
                    model_name=loaded_model.name,
                )
            except Exception as e:
                logger.error("Failed to attach inference data: %s", e, exc_info=True)
                continue

            # Enqueue for downstream dispatchers/visualization
            try:
                pred_queue.put(stream_data, timeout=1)
            except std_queue.Full:
                logger.debug("Prediction queue is full; dropping result")
                continue
    finally:
        # https://docs.python.org/3/library/multiprocessing.html#all-start-methods
        # section: Joining processes that use queues
        # Call cancel_join_thread() to prevent the parent process from blocking
        # indefinitely when joining child processes that used this queue. This avoids potential
        # deadlocks if the queue's background thread adds more items during the flush.
        if pred_queue is not None:
            logger.debug("Cancelling the pred_queue join thread to allow inference process to exit")
            pred_queue.cancel_join_thread()

        log_threads(log_level=logging.DEBUG)
        logger.info("Stopped inference routine")


def inference_routine(
    frame_queue: mp.Queue,
    pred_queue: mp.Queue,
    stop_event: EventClass,
    model_reload_event: EventClass,
    shm_name: str,
    shm_lock: Lock,
) -> None:
    """Load frames from the frame queue, run inference then inject the result into the predictions queue"""
    suppress_child_shutdown_signals()

    asyncio.run(
        _inference_loop(
            frame_queue=frame_queue,
            pred_queue=pred_queue,
            stop_event=stop_event,
            model_reload_event=model_reload_event,
            shm_name=shm_name,
            shm_lock=shm_lock,
        )
    )
