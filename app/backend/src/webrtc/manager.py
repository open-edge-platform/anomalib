# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import queue
from typing import Any

from aiortc import RTCPeerConnection, RTCSessionDescription

from pydantic_models.webrtc import Answer, InputData, Offer
from webrtc.stream import InferenceVideoStreamTrack

logger = logging.getLogger(__name__)


class WebRTCManager:
    """Manager for handling WebRTC connections."""

    def __init__(self, stream_queue: queue.Queue) -> None:
        self._pcs: dict[str, RTCPeerConnection] = {}
        self._input_data: dict[str, Any] = {}
        self._stream_queue = stream_queue

    async def handle_offer(self, offer: Offer) -> Answer:
        """Create an SDP offer for a new WebRTC connection."""
        pc = RTCPeerConnection()
        self._pcs[offer.webrtc_id] = pc

        # Add video track
        track = InferenceVideoStreamTrack(self._stream_queue)
        pc.addTrack(track)

        @pc.on("connectionstatechange")
        async def connection_state_change() -> None:
            if pc.connectionState in ["failed", "closed"]:
                await self.cleanup_connection(offer.webrtc_id)

        # Set remote description from client's offer
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return Answer(sdp=pc.localDescription.sdp, type=pc.localDescription.type)

    def set_input(self, data: InputData) -> None:
        """Set input data for specific WebRTC connection"""
        self._input_data[data.webrtc_id] = {
            "conf_threshold": data.conf_threshold,
            "updated_at": asyncio.get_event_loop().time(),
        }

    async def cleanup_connection(self, webrtc_id: str) -> None:
        """Clean up a specific WebRTC connection by its ID."""
        if webrtc_id in self._pcs:
            logger.debug("Cleaning up connection: %s", webrtc_id)
            pc = self._pcs.pop(webrtc_id)
            await pc.close()
            logger.debug("Connection %s successfully closed.", webrtc_id)
            self._input_data.pop(webrtc_id, None)

    async def cleanup(self) -> None:
        """Clean up all connections"""
        for pc in list(self._pcs.values()):
            await pc.close()
        self._pcs.clear()
        self._input_data.clear()
