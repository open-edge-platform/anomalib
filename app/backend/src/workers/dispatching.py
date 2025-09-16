# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import multiprocessing as mp
import queue
import time
from multiprocessing.synchronize import Event as EventClass

from entities.stream_data import StreamData
from pydantic_models import Sink, SinkType
from services import DispatchService
from services.dispatchers import Dispatcher

logger = logging.getLogger(__name__)


def dispatching_routine(
    pred_queue: mp.Queue,
    rtc_stream_queue: queue.Queue,
    stop_event: EventClass,
) -> None:
    """Pull predictions from the queue and dispatch them to the configured outputs and WebRTC visualization stream."""
    # Avoid circular import
    from api.dependencies import get_active_pipeline_service  # noqa: PLC0415

    active_pipeline_service = get_active_pipeline_service()

    prev_sink_config: Sink | None = None
    destinations: list[Dispatcher] = []

    try:
        while not stop_event.is_set():
            sink_config = active_pipeline_service.get_sink_config()

            if sink_config.sink_type == SinkType.DISCONNECTED:
                logger.debug("No sink available... retrying in 1 second")
                time.sleep(1)
                continue

            if not prev_sink_config or sink_config != prev_sink_config:
                logger.debug(f"Sink config changed from {prev_sink_config} to {sink_config}")
                destinations = DispatchService.get_destinations(output_configs=[sink_config])
                prev_sink_config = copy.deepcopy(sink_config)

            # Read from the queue
            try:
                stream_data: StreamData = pred_queue.get(timeout=1)
            except queue.Empty:
                logger.debug("Nothing to dispatch yet")
                continue

            if stream_data.inference_data is None:
                logger.error("Missing inference data in stream_data; skipping dispatch")
                continue

            inference_data = stream_data.inference_data
            if inference_data is None:
                logger.error("No inference data available")
                continue

            image_with_visualization = inference_data.visualized_prediction
            prediction = inference_data.prediction
            # Postprocess and dispatch results
            for destination in destinations:
                destination.dispatch(
                    original_image=stream_data.frame_data,
                    image_with_visualization=image_with_visualization,
                    predictions=prediction,
                )

            # Dispatch to WebRTC stream
            try:
                rtc_stream_queue.put(image_with_visualization, block=False)
            except queue.Full:
                logger.debug("Visualization queue is full; skipping")
    finally:
        logger.info("Stopped dispatching routine")
