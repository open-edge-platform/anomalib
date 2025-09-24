# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import logging
import multiprocessing as mp
import queue as std_queue
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.synchronize import Lock
from typing import Any

import cv2
import numpy as np

from services import ModelService
from services.metrics_service import MetricsService
from services.model_service import LoadedModel
from utils import log_threads, suppress_child_shutdown_signals

logger = logging.getLogger(__name__)


async def _inference_loop(  # noqa: C901, PLR0912, PLR0915
    frame_queue: mp.Queue,
    pred_queue: mp.Queue,
    stop_event: EventClass,
    model_reload_event: EventClass,
    shm_name: str,
    shm_lock: Lock,
) -> None:
    # Local imports to avoid circular dependencies in workers
    from db import get_async_db_session_ctx  # noqa: PLC0415
    from entities.stream_data import InferenceData, StreamData  # noqa: PLC0415
    from repositories import PipelineRepository  # noqa: PLC0415

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

            # Build visualization: overlay anomaly heatmap and label/score
            try:
                vis_frame: np.ndarray = frame.copy()

                # Decode anomaly map from base64 → numpy image
                try:
                    anomaly_png_bytes = base64.b64decode(prediction_response.anomaly_map)
                    anomaly_np = np.frombuffer(anomaly_png_bytes, dtype=np.uint8)
                    anomaly_img = cv2.imdecode(anomaly_np, cv2.IMREAD_UNCHANGED)
                except Exception:
                    anomaly_img = None

                # If available, colorize, resize and overlay the anomaly map (thresholded)
                if anomaly_img is not None:
                    try:
                        if anomaly_img.ndim == 3 and anomaly_img.shape[2] > 1:
                            # Convert to single channel if needed
                            anomaly_gray = cv2.cvtColor(anomaly_img, cv2.COLOR_BGR2GRAY)
                        else:
                            anomaly_gray = anomaly_img

                        # Ensure uint8 for colormap, then apply heatmap
                        if anomaly_gray.dtype != np.uint8:
                            anomaly_gray = anomaly_gray.astype(np.uint8)

                        # Build heatmap and a 0.5-threshold mask (0.5 * 255 == 128)
                        heatmap = cv2.applyColorMap(anomaly_gray, cv2.COLORMAP_JET)
                        heatmap_resized = cv2.resize(heatmap, (vis_frame.shape[1], vis_frame.shape[0]))

                        mask_gray = cv2.resize(anomaly_gray, (vis_frame.shape[1], vis_frame.shape[0]))
                        threshold_value = 128
                        mask_bool = mask_gray >= threshold_value

                        # Create masked overlay (zeros outside mask)
                        masked_heatmap = np.zeros_like(heatmap_resized)
                        try:
                            masked_heatmap[mask_bool] = heatmap_resized[mask_bool]
                        except Exception as e:
                            logger.debug("Failed to apply heatmap mask: %s", e)

                        # More transparent overlay
                        alpha = 0.25
                        vis_frame = cv2.addWeighted(vis_frame, 1.0, masked_heatmap, alpha, 0)
                    except Exception as e:
                        logger.debug("Failed to overlay heatmap: %s", e)

                # Draw larger, clearer label text with background
                label_text = f"{prediction_response.label.value} ({prediction_response.score:.3f})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2.0
                thickness = 3
                (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
                x, y = 10, 20 + text_h  # top-left with margin
                # Background rectangle for readability
                cv2.rectangle(vis_frame, (x - 8, y - text_h - 8), (x - 8 + text_w + 16, y + 8), (0, 0, 0), -1)
                cv2.putText(vis_frame, label_text, (x, y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
            except Exception as e:
                logger.debug("Failed to create visualization: %s", e)
                vis_frame = frame

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
