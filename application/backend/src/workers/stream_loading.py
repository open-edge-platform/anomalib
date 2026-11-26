# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import copy
import queue
from typing import TYPE_CHECKING

import loguru
from loguru import logger

from pydantic_models import Source, SourceType
from services import ActivePipelineService, VideoStreamService
from workers.base import BaseProcessWorker

if TYPE_CHECKING:
    import multiprocessing as mp
    from multiprocessing.synchronize import Condition as ConditionClass
    from multiprocessing.synchronize import Event as EventClass

    from entities.stream_data import StreamData
    from entities.video_stream import VideoStream


class StreamLoader(BaseProcessWorker):
    """A process that loads frames from the video stream and injects them into the frame queue."""

    ROLE = "StreamLoader"

    def __init__(
        self,
        frame_queue: mp.Queue,
        stop_event: EventClass,
        config_changed_condition: ConditionClass,
        logger_: loguru.Logger | None = None,
    ) -> None:
        super().__init__(stop_event=stop_event, queues_to_cancel=[frame_queue], logger_=logger_)
        self._frame_queue = frame_queue
        self._config_changed_condition = config_changed_condition

        self._active_pipeline_service: ActivePipelineService | None = None
        self._prev_source_config: Source | None = None
        self._video_stream: VideoStream | None = None

    def _reset_stream_if_needed(self, source_config: Source) -> None:
        if self._prev_source_config is None or source_config != self._prev_source_config:
            logger.debug(f"Source configuration changed from {self._prev_source_config} to {source_config}")
            if self._video_stream is not None:
                self._video_stream.release()
            self._video_stream = VideoStreamService.get_video_stream(input_config=source_config)
            self._prev_source_config = copy.deepcopy(source_config)

    @logger.catch()
    async def run_loop(self) -> None:
        self._active_pipeline_service = await ActivePipelineService.create(
            config_changed_condition=self._config_changed_condition,
            start_daemon=True,
        )

        while not self.should_stop():
            try:
                source_config = self._active_pipeline_service.source_config

                if source_config.source_type == SourceType.DISCONNECTED:
                    logger.trace("No source available... retrying in 1 second")
                    await asyncio.sleep(1)
                    continue

                self._reset_stream_if_needed(source_config)

                if self._video_stream is None:
                    logger.trace("No video stream available, retrying in 1 second...")
                    await asyncio.sleep(1)
                    continue

                # Acquire a frame and enqueue it
                try:
                    stream_data = self._video_stream.get_data()
                    if stream_data is not None:
                        _enqueue_frame_with_retry(
                            self._frame_queue,
                            stream_data,
                            self._video_stream.is_real_time(),
                            self._stop_event,
                        )
                    else:
                        await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error acquiring frame. Details: `{str(e)}`")
                    await asyncio.sleep(1)
                    continue
            except Exception as e:
                logger.error(f"Error acquiring frame. Details: `{str(e)}`")
                continue

    def teardown(self) -> None:
        if self._video_stream is not None:
            logger.debug("Releasing video stream...")
            self._video_stream.release()


def _enqueue_frame_with_retry(
    frame_queue: mp.Queue,
    payload: StreamData,
    is_real_time: bool,
    stop_event: EventClass,
) -> None:
    """Enqueue frame with retry logic for non-real-time streams"""
    while not stop_event.is_set():
        try:
            frame_queue.put(payload, timeout=1)
            break
        except queue.Full:
            if is_real_time:
                frame_queue.get(timeout=0.01)  # Discard oldest frame
                logger.debug("Frame queue is full, discarded oldest frame for real-time stream")
                break
            logger.debug("Frame queue is full, retrying...")
