# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import queue
import time
from collections.abc import AsyncIterator
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.dependencies import get_scheduler
from api.endpoints import API_PREFIX
from core import Scheduler

router = APIRouter(
    prefix=f"{API_PREFIX}/stream",
    tags=["stream"],
)

JPEG_QUALITY = 85
FRAME_TIMEOUT_SEC = 0.5
STREAM_BOUNDARY = "frame"
MAX_FPS = 10
FRAME_INTERVAL_SEC = 1.0 / MAX_FPS


def encode_frame_to_jpeg(frame: np.ndarray) -> bytes | None:
    """Encode an RGB frame to JPEG bytes.

    Args:
        frame (np.ndarray): RGB frame to encode.

    Returns:
        bytes | None: Encoded JPEG bytes, or None if encoding fails.
    """
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    success, jpeg = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not success:
        return None
    return jpeg.tobytes()


async def generate_mjpeg_stream(stream_queue: queue.Queue) -> AsyncIterator[bytes]:
    """Yield MJPEG frames from the stream queue.

    Args:
        stream_queue (queue.Queue): Queue containing numpy RGB frames.

    Returns:
        AsyncIterator[bytes]: Multipart MJPEG byte chunks.
    """
    last_frame: np.ndarray | None = None
    last_emit_at: float | None = None

    while True:
        try:
            frame = await asyncio.to_thread(stream_queue.get, True, FRAME_TIMEOUT_SEC)
            while True:
                try:
                    frame = stream_queue.get_nowait()
                except queue.Empty:
                    break
            last_frame = frame
        except queue.Empty:
            if last_frame is None:
                await asyncio.sleep(0.1)
                continue
            frame = last_frame

        if frame is None:
            await asyncio.sleep(0.1)
            continue

        if last_emit_at is not None:
            elapsed = time.monotonic() - last_emit_at
            if elapsed < FRAME_INTERVAL_SEC:
                await asyncio.sleep(FRAME_INTERVAL_SEC - elapsed)

        jpeg_bytes = await asyncio.to_thread(encode_frame_to_jpeg, frame)
        if jpeg_bytes is None:
            continue
        last_emit_at = time.monotonic()

        yield (f"--{STREAM_BOUNDARY}\r\n".encode() + b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n")


@router.get("")
async def stream(scheduler: Annotated[Scheduler, Depends(get_scheduler)]) -> StreamingResponse:
    """Stream the active pipeline output as MJPEG.

    Args:
        scheduler (Scheduler): Global scheduler providing the stream queue.

    Returns:
        StreamingResponse: Multipart MJPEG response.
    """
    return StreamingResponse(
        generate_mjpeg_stream(scheduler.rtc_stream_queue),
        media_type=f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}",
    )
