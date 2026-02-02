# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from api.dependencies import get_scheduler
from api.endpoints import API_PREFIX
from core import Scheduler

router = APIRouter(
    prefix=f"{API_PREFIX}/stream",
    tags=["stream"],
)

STREAM_BOUNDARY = "frame"


async def generate_mjpeg_stream(scheduler: Scheduler, request: Request) -> AsyncIterator[bytes]:
    """Yield MJPEG frames from the broadcaster.

    Args:
        scheduler (Scheduler): Scheduler containing the MJPEG broadcaster.
        request (Request): FastAPI request used to detect client disconnects.

    Yields:
        bytes: Multipart MJPEG byte chunks.
    """
    last_seen_id = 0
    logger.info("MJPEG stream started")

    try:
        while True:
            if scheduler.mp_stop_event.is_set():
                logger.info("Shutdown requested; stopping MJPEG stream")
                break

            if await request.is_disconnected():
                logger.info("Client disconnected")
                break

            jpeg_bytes, last_seen_id = await scheduler.mjpeg_broadcaster.get_jpeg(last_seen_id, timeout=1.0)
            if jpeg_bytes is None:
                continue

            yield (f"--{STREAM_BOUNDARY}\r\n".encode() + b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n")
    except asyncio.CancelledError:
        logger.warning("MJPEG stream cancelled")
    finally:
        logger.info("MJPEG stream stopped")


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
    return StreamingResponse(
        generate_mjpeg_stream(scheduler, request),
        media_type=f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}",
    )
