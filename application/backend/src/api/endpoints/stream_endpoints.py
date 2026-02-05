# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from loguru import logger

from api.dependencies import get_scheduler
from api.endpoints import API_PREFIX
from core import Scheduler
from settings import get_settings

router = APIRouter(
    prefix=f"{API_PREFIX}/stream",
    tags=["stream"],
)

STREAM_BOUNDARY = "frame"
_active_stream_clients = 0
_stream_clients_lock = asyncio.Lock()


async def generate_mjpeg_stream(scheduler: Scheduler, request: Request) -> AsyncIterator[bytes]:
    """Yield MJPEG frames from the broadcaster.

    Args:
        scheduler (Scheduler): Scheduler containing the MJPEG broadcaster.
        request (Request): FastAPI request used to detect client disconnects.

    Yields:
        bytes: Multipart MJPEG byte chunks.
    """
    last_seen_id = 0
    stream_id = _active_stream_clients
    logger.info(f"MJPEG stream started ({stream_id})")

    try:
        while True:
            if scheduler.mp_stop_event.is_set():
                logger.info("Shutdown requested; stopping MJPEG stream")
                break

            if await request.is_disconnected():
                logger.info("Client disconnected")
                break

            jpeg_bytes, last_seen_id = await scheduler.mjpeg_broadcaster.get_jpeg(last_seen_id, timeout=0.1)
            if jpeg_bytes is None:
                continue

            yield (f"--{STREAM_BOUNDARY}\r\n".encode() + b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n")
    except asyncio.CancelledError:
        logger.debug(f"MJPEG stream cancelled ({stream_id})")
    finally:
        logger.info(f"MJPEG stream stopped ({stream_id})")


@router.get("")
async def stream(
    scheduler: Annotated[Scheduler, Depends(get_scheduler)],
    request: Request,
) -> StreamingResponse:
    """Stream the active pipeline output as MJPEG.

    Args:
        scheduler (Scheduler): Global scheduler providing the MJPEG broadcaster.
        request (Request): FastAPI request for disconnect detection.

    Returns:
        StreamingResponse: Multipart MJPEG response.
    """
    global _active_stream_clients  # noqa: PLW0603
    settings = get_settings()
    async with _stream_clients_lock:
        if _active_stream_clients >= settings.stream_max_clients:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Too many stream clients")
        _active_stream_clients += 1

    async def guarded_stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in generate_mjpeg_stream(scheduler, request):
                yield chunk
        finally:
            global _active_stream_clients  # noqa: PLW0603
            async with _stream_clients_lock:
                _active_stream_clients = max(0, _active_stream_clients - 1)

    return StreamingResponse(
        guarded_stream(),
        media_type=f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}",
    )
