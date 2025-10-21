# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import multiprocessing as mp
import queue
from multiprocessing.synchronize import Condition as ConditionClass
from multiprocessing.synchronize import Event as EventClass

from loguru import logger

from entities.stream_data import StreamData
from pydantic_models import Sink, SinkType
from services import ActivePipelineService, DispatchService
from services.dispatchers import Dispatcher
from workers.base import BaseThreadWorker


class DispatchingWorker(BaseThreadWorker):
    """
    A thread that pulls predictions from the queue and dispatches them to the configured outputs
    and WebRTC visualization stream.
    """

    ROLE = "Dispatching"

    def __init__(
        self,
        pred_queue: mp.Queue,
        rtc_stream_queue: queue.Queue,
        stop_event: EventClass,
        active_config_changed_condition: ConditionClass,
    ) -> None:
        super().__init__(stop_event=stop_event)
        self._pred_queue = pred_queue
        self._rtc_stream_queue = rtc_stream_queue

        self._active_config_changed_condition = active_config_changed_condition

        self._prev_sink_config: Sink | None = None
        self._destinations: list[Dispatcher] = []

    def _reset_sink_if_needed(self, sink_config: Sink) -> None:
        if not self._prev_sink_config or sink_config != self._prev_sink_config:
            logger.debug(f"Sink config changed from {self._prev_sink_config} to {sink_config}")
            self._destinations = DispatchService.get_destinations(output_configs=[sink_config])
            self._prev_sink_config = copy.deepcopy(sink_config)

    async def run_loop(self) -> None:
        active_pipeline_service = await ActivePipelineService.create(
            self._active_config_changed_condition, start_daemon=True
        )

        while not self.should_stop():
            # Exit if parent process died (if ever run as a process)
            parent_process = mp.parent_process()
            if parent_process is not None and not parent_process.is_alive():
                break
            sink_config = active_pipeline_service.get_sink_config()

            if sink_config.sink_type == SinkType.DISCONNECTED:
                logger.trace("No sink available... retrying in 1 second")
                await asyncio.sleep(1)
                continue

            self._reset_sink_if_needed(sink_config)

            # Read from the queue
            try:
                stream_data: StreamData = self._pred_queue.get(timeout=1)
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
            async with asyncio.TaskGroup() as task_group:
                for destination in self._destinations:
                    task_group.create_task(
                        destination.dispatch(
                            original_image=stream_data.frame_data,
                            image_with_visualization=image_with_visualization,
                            predictions=prediction,
                        )
                    )
                # Dispatch to WebRTC stream
                try:
                    self._rtc_stream_queue.put(image_with_visualization, block=False)
                except queue.Full:
                    logger.debug("Visualization queue is full; skipping")
