# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import queue

import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame
from loguru import logger

FALLBACK_FRAME = np.full((64, 64, 3), 16, dtype=np.uint8)


class InferenceVideoStreamTrack(VideoStreamTrack):
    """A video stream track that provides frames with inference results over WebRTC."""

    def __init__(self, stream_queue: queue.Queue[np.ndarray]):
        super().__init__()
        self._stream_queue = stream_queue
        self._last_frame: np.ndarray | None = None

    async def recv(self) -> VideoFrame:
        """
        Asynchronously receive the next video frame from the internal queue.

        This coroutine attempts to obtain a frame from ``self._stream_queue`` with a
        500ms timeout. If a new frame is received, it is cached in ``self._last_frame``.
        If the queue is empty and no cached frame exists, the method uses the
        ``FALLBACK_FRAME``, a small, 64 x 64, dark gray numpy array
        representing a video frame.

        The received or fallback frame is wrapped in a ``VideoFrame`` object, with
        its presentation timestamp (``pts``) and time base attached.

        Behavior:
            - Pulls frames from ``_stream_queue`` using ``asyncio.to_thread`` (timeout: 500ms).
            - On timeout, returns last cached frame if available.
            - If no cached frame exists, returns ``FALLBACK_FRAME``.
            - Ensures robust streaming when new frames are intermittently missing.

        Returns:
            aiortc.VideoFrame:
                Video frame object containing image data, presentation timestamp (``pts``),
                and time base.

        Raises:
            Exception:
                Logs and propagates any errors during retrieval or conversion.

        Notes:
            - Uses ``asyncio.to_thread`` to prevent blocking the event loop
              when calling the synchronous ``queue.Queue.get`` method.
            - Ensures resilience of streaming by falling back to cached or
              dummy frames in case of delayed or missing input.

        """
        pts, time_base = await self.next_timestamp()

        try:
            try:
                logger.trace("Getting the frame from the stream_queue...")
                frame_data = await asyncio.to_thread(self._stream_queue.get, True, 0.5)  # wait for 500ms
                self._last_frame = frame_data  # cache the successful frame
            except queue.Empty:
                logger.trace("Empty queue. Using the last frame...")
                if self._last_frame is None:
                    frame_data = FALLBACK_FRAME
                else:
                    frame_data = self._last_frame

            logger.trace("Received the frame from the stream_queue.")

            # Convert numpy array to VideoFrame
            frame = VideoFrame.from_ndarray(frame_data, format="rgb24")
            frame.pts = pts
            frame.time_base = time_base
            return frame
        except Exception as e:
            logger.error(f"Error in recv: {e}")
            raise
