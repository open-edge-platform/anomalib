# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import queue
from typing import Any

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcconfiguration import RTCConfiguration, RTCIceServer
from loguru import logger

from pydantic_models.webrtc import Answer, InputData, Offer
from webrtc.stream import InferenceVideoStreamTrack


class WebRTCManager:
    """Manager for handling WebRTC connections."""

    def __init__(self, stream_queue: queue.Queue) -> None:
        self._peer_connections: dict[str, RTCPeerConnection] = {}
        self._input_data: dict[str, Any] = {}
        self._stream_queue = stream_queue

    async def handle_offer(self, offer: Offer) -> Answer:
        """Create an SDP offer for a new WebRTC connection."""
        try:
            # Configure RTCPeerConnection with proper ICE servers
            pc = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]))
            self._peer_connections[offer.webrtc_id] = pc

            # Add video track
            track = InferenceVideoStreamTrack(self._stream_queue)
            pc.addTrack(track)

            @pc.on("connectionstatechange")
            async def connection_state_change() -> None:
                if pc.connectionState in {"failed", "closed"}:
                    await self.cleanup_connection(offer.webrtc_id)

            # Validate offer before processing
            if not offer.sdp or not offer.type:
                raise ValueError("Invalid offer: missing SDP or type")

            # Set remote description from client's offer
            await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))

            # Create answer with proper configuration
            answer = await pc.createAnswer()

            # Set local description with error handling
            try:
                await pc.setLocalDescription(answer)
            except Exception as e:
                logger.error(f"Failed to set local description: {e}")
                # Try with a simpler answer configuration
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

            # Ensure local description is set
            if pc.localDescription is None:
                raise RuntimeError("Failed to create local description")

            return Answer(sdp=pc.localDescription.sdp, type=pc.localDescription.type)
        except Exception as e:
            logger.error(f"Error in handle_offer: {e}", exc_info=True)
            # Clean up on error
            if offer.webrtc_id in self._peer_connections:
                await self.cleanup_connection(offer.webrtc_id)
            raise

    def set_input(self, data: InputData) -> None:
        """Set input data for specific WebRTC connection"""
        self._input_data[data.webrtc_id] = {
            "conf_threshold": data.conf_threshold,
            "updated_at": asyncio.get_event_loop().time(),
        }

    async def cleanup_connection(self, webrtc_id: str) -> None:
        """Clean up a specific WebRTC connection by its ID."""
        if webrtc_id in self._peer_connections:
            logger.debug(f"Cleaning up connection: {webrtc_id}")
            pc = self._peer_connections.pop(webrtc_id)
            await pc.close()
            logger.debug(f"Connection {webrtc_id} successfully closed.")
            self._input_data.pop(webrtc_id, None)

    async def cleanup(self) -> None:
        """Clean up all connections"""
        for pc in list(self._peer_connections.values()):
            await pc.close()
        self._peer_connections.clear()
        self._input_data.clear()
