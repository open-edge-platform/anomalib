# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MJPEG frame broadcaster for multiple async consumers."""

from __future__ import annotations

import asyncio

from loguru import logger


class MJPEGBroadcaster:
    """Broadcast JPEG bytes to multiple async consumers using asyncio.Condition."""

    def __init__(self) -> None:
        self._jpeg_bytes: bytes | None = None
        self._frame_id: int = 0
        self._condition: asyncio.Condition | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def initialize(self, loop: asyncio.AbstractEventLoop) -> None:
        """Initialize the broadcaster with the main event loop.

        Args:
            loop (asyncio.AbstractEventLoop): Event loop used to publish frames.
        """
        self._loop = loop
        self._condition = asyncio.Condition()
        logger.info("MJPEGBroadcaster initialized")

    def publish_threadsafe(self, jpeg_bytes: bytes) -> None:
        """Publish JPEG bytes from a sync thread.

        Args:
            jpeg_bytes (bytes): JPEG encoded frame bytes.
        """
        if self._loop is None or self._condition is None:
            return
        if self._loop.is_closed() or not self._loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._publish(jpeg_bytes), self._loop)

    async def _publish(self, jpeg_bytes: bytes) -> None:
        """Publish JPEG bytes and notify all waiting consumers.

        Args:
            jpeg_bytes (bytes): JPEG encoded frame bytes.
        """
        if self._condition is None:
            raise RuntimeError("MJPEGBroadcaster not initialized")

        async with self._condition:
            self._jpeg_bytes = jpeg_bytes
            self._frame_id += 1
            self._condition.notify_all()

    async def get_jpeg(self, last_seen_id: int, timeout: float = 0.5) -> tuple[bytes | None, int]:  # noqa: ASYNC109
        """Wait for new JPEG bytes or timeout.

        Args:
            last_seen_id (int): Last frame id seen by the consumer.
            timeout (float): Time to wait for a new frame. Defaults to 0.5.

        Returns:
            tuple[bytes | None, int]: JPEG bytes and new frame id or (None, last_seen_id) on timeout.
        """
        if self._condition is None:
            return None, last_seen_id
        try:
            async with self._condition:
                await asyncio.wait_for(
                    self._condition.wait_for(lambda: self._frame_id != last_seen_id),
                    timeout=timeout,
                )
                return self._jpeg_bytes, self._frame_id
        except TimeoutError:
            return None, last_seen_id

    def shutdown(self) -> None:
        """Disable publishing and detach from the event loop."""
        self._condition = None
        self._loop = None
