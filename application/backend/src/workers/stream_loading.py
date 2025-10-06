# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import logging
import multiprocessing as mp
import queue
from multiprocessing.synchronize import Condition as ConditionClass
from multiprocessing.synchronize import Event as EventClass

from entities.stream_data import StreamData
from entities.video_stream import VideoStream
from pydantic_models import Source, SourceType
from services import ActivePipelineService, VideoStreamService
from utils import log_threads, suppress_child_shutdown_signals

logger = logging.getLogger(__name__)


async def _acquisition_loop(  # noqa: C901
    frame_queue: mp.Queue, stop_event: EventClass, config_changed_condition: ConditionClass, cleanup: bool = True
) -> None:
    active_pipeline_service = await ActivePipelineService.create(config_changed_condition, start_daemon=True)
    prev_source_config: Source | None = None
    video_stream: VideoStream | None = None

    try:
        while not stop_event.is_set():
            # Exit if parent process died
            parent_process = mp.parent_process()
            if parent_process is not None and not parent_process.is_alive():
                break
            try:
                source_config = active_pipeline_service.get_source_config()

                if source_config.source_type == SourceType.DISCONNECTED:
                    logger.debug("No source available... retrying in 1 second")
                    await asyncio.sleep(1)
                    continue

                # Reset the video stream if the configuration has changed
                if prev_source_config is None or source_config != prev_source_config:
                    logger.debug(f"Source configuration changed from {prev_source_config} to {source_config}")
                    if video_stream is not None:
                        video_stream.release()
                    video_stream = VideoStreamService.get_video_stream(input_config=source_config)
                    prev_source_config = copy.deepcopy(source_config)

                if video_stream is None:
                    logger.debug("No video stream available, retrying in 1 second...")
                    await asyncio.sleep(1)
                    continue

                # Acquire a frame and enqueue it
                try:
                    stream_data = video_stream.get_data()
                    if stream_data is not None:
                        _enqueue_frame_with_retry(frame_queue, stream_data, video_stream.is_real_time(), stop_event)
                    else:
                        await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error acquiring frame: {e}")
                    continue
            except Exception as e:
                logger.error(f"Error acquiring frame: {e}")
                continue
    finally:
        if cleanup:
            _cleanup_resources(frame_queue, video_stream)
        logger.info("Stopped stream acquisition")


def frame_acquisition_routine(
    frame_queue: mp.Queue, stop_event: EventClass, config_changed_condition: ConditionClass, cleanup: bool = True
) -> None:
    """Load frames from the video stream and inject them into the frame queue"""
    suppress_child_shutdown_signals()
    asyncio.run(_acquisition_loop(frame_queue, stop_event, config_changed_condition, cleanup))


def _enqueue_frame_with_retry(
    frame_queue: mp.Queue, payload: StreamData, is_real_time: bool, stop_event: EventClass
) -> None:
    """Enqueue frame with retry logic for non-real-time streams"""
    while not stop_event.is_set():
        try:
            frame_queue.put(payload, timeout=1)
            break
        except queue.Full:
            if is_real_time:
                logger.debug("Frame queue is full, skipping frame")
                break
            logger.debug("Frame queue is full, retrying...")


def _cleanup_resources(frame_queue: mp.Queue, video_stream: VideoStream | None) -> None:
    """Clean up video stream and frame queue resources"""
    if video_stream is not None:
        logger.debug("Releasing video stream...")
        video_stream.release()

    # https://docs.python.org/3/library/multiprocessing.html#all-start-methods
    # section: Joining processes that use queues
    # Call cancel_join_thread() to prevent the parent process from blocking
    # indefinitely when joining child processes that used this queue. This avoids potential
    # deadlocks if the queue's background thread adds more items during the flush.
    if frame_queue is not None:
        logger.debug("Cancelling the frame_queue join thread to allow inference process to exit")
        frame_queue.cancel_join_thread()

    log_threads(log_level=logging.DEBUG)
