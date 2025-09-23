# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""WebRTC API Endpoints"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.exceptions import HTTPException

from api.dependencies import get_webrtc_manager as get_webrtc
from pydantic_models.webrtc import Answer, InputData, Offer
from webrtc.manager import WebRTCManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/webrtc", tags=["WebRTC"])


@router.post(
    "/offer",
    responses={
        status.HTTP_200_OK: {"description": "WebRTC Answer"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Internal Server Error"},
    },
)
async def create_webrtc_offer(offer: Offer, webrtc_manager: Annotated[WebRTCManager, Depends(get_webrtc)]) -> Answer:
    """Create a WebRTC offer"""
    try:
        return await webrtc_manager.handle_offer(offer)
    except Exception as e:
        logger.error("Error processing WebRTC offer: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post(
    "/input_hook",
    responses={
        status.HTTP_200_OK: {"description": "WebRTC input data updated"},
    },
)
async def webrtc_input_hook(data: InputData, webrtc_manager: Annotated[WebRTCManager, Depends(get_webrtc)]) -> None:
    """Update webrtc input with user data"""
    webrtc_manager.set_input(data)
