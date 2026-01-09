# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import queue
from dataclasses import dataclass
from typing import Any

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.rtcicetransport import RTCIceCandidate
from loguru import logger

from pydantic_models.webrtc import Answer, InputData, Offer

from .sdp_handler import SDPHandler
from .stream import InferenceVideoStreamTrack


@dataclass(frozen=True)
class WebRTCSettings:
    config: RTCConfiguration
    advertise_ip: str | None = None


class WebRTCManager:
    """Manager for handling WebRTC connections."""

    async def _wait_for_ice_gathering_complete(self, pc: RTCPeerConnection, timeout_s: float = 10.0) -> None:
        """Wait until ICE gathering is complete.

        This project does not implement trickle ICE (candidate exchange after SDP),
        so we wait for ICE gathering to complete to include candidates in the SDP.
        """

        if pc.iceGatheringState == "complete":
            return

        done = asyncio.Event()

        @pc.on("icegatheringstatechange")
        def ice_gathering_state_change() -> None:  # pragma: no cover (callback)
            print(f"DEBUG_PRINT: ICE gathering state changed to: {pc.iceGatheringState}")
            logger.debug(f"DEBUG: ICE gathering state changed to: {pc.iceGatheringState}")
            if pc.iceGatheringState == "complete":
                done.set()

        print(f"DEBUG_PRINT: Waiting for ICE gathering to complete (timeout={timeout_s}s)...")
        logger.debug(f"DEBUG: Waiting for ICE gathering to complete (timeout={timeout_s}s)...")
        try:
            async with asyncio.timeout(timeout_s):
                await done.wait()
            print("DEBUG_PRINT: ICE gathering completed successfully.")
            logger.debug("DEBUG: ICE gathering completed successfully.")
        except TimeoutError:
            print("DEBUG_PRINT: ICE gathering timed out!")
            logger.debug("DEBUG: ICE gathering timed out!")

    def __init__(self, stream_queue: queue.Queue, settings: WebRTCSettings, sdp_handler: SDPHandler) -> None:
        self._pcs: dict[str, RTCPeerConnection] = {}
        self._input_data: dict[str, Any] = {}
        self._stream_queue = stream_queue
        self._settings = settings
        self._sdp_handler = sdp_handler

    async def handle_offer(self, offer: Offer) -> Answer:
        """Create an SDP offer for a new WebRTC connection."""
        pc = RTCPeerConnection(configuration=self._settings.config)
        self._pcs[offer.webrtc_id] = pc

        @pc.on("icecandidate")
        def ice_candidate(candidate: RTCIceCandidate | None) -> None:  # pragma: no cover (callback)
            # This app does not implement trickle ICE, but logging candidates is very useful for debugging.
            if candidate is None:
                msg = f"WebRTC {offer.webrtc_id}: ICE gathering complete (end-of-candidates)."
                print(f"DEBUG_PRINT: {msg}")
                logger.debug(f"DEBUG: {msg}")
                logger.debug(msg)
                return

            msg = (
                f"WebRTC {offer.webrtc_id}: ICE candidate gathered: "
                f"{candidate.protocol} {candidate.ip}:{candidate.port} {candidate.transport} type={candidate.type}"
            )
            print(f"DEBUG_PRINT: {msg}")
            logger.debug(f"DEBUG: {msg}")
            logger.debug(msg)

        @pc.on("iceconnectionstatechange")
        async def ice_connection_state_change() -> None:  # pragma: no cover (callback)
            logger.debug("WebRTC {}: ICE connection state: {}", offer.webrtc_id, pc.iceConnectionState)

        # Add video track
        track = InferenceVideoStreamTrack(self._stream_queue)
        pc.addTrack(track)

        @pc.on("connectionstatechange")
        async def connection_state_change() -> None:
            logger.debug("WebRTC {}: Connection state: {}", offer.webrtc_id, pc.connectionState)
            if pc.connectionState in {"failed", "closed"}:
                await self.cleanup_connection(offer.webrtc_id)

        # Set remote description from client's offer
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await self._wait_for_ice_gathering_complete(pc)
        logger.debug(
            "WebRTC {}: LocalDescription ready. iceGatheringState={}, len(sdp)={}",
            offer.webrtc_id,
            pc.iceGatheringState,
            len(pc.localDescription.sdp) if pc.localDescription and pc.localDescription.sdp else 0,
        )

        # Mangle SDP if public IP is configured
        sdp = pc.localDescription.sdp
        if self._settings.advertise_ip:
            sdp = await self._sdp_handler.mangle_sdp(sdp, self._settings.advertise_ip)

        return Answer(sdp=sdp, type=pc.localDescription.type)

    def set_input(self, data: InputData) -> None:
        """Set input data for specific WebRTC connection"""
        self._input_data[data.webrtc_id] = {
            "conf_threshold": data.conf_threshold,
            "updated_at": asyncio.get_event_loop().time(),
        }

    async def cleanup_connection(self, webrtc_id: str) -> None:
        """Clean up a specific WebRTC connection by its ID."""
        if webrtc_id in self._pcs:
            logger.debug("Cleaning up connection: {}", webrtc_id)
            pc = self._pcs.pop(webrtc_id)
            await pc.close()
            logger.debug("Connection {} successfully closed.", webrtc_id)
            self._input_data.pop(webrtc_id, None)

    async def cleanup(self) -> None:
        """Clean up all connections"""
        for pc in list(self._pcs.values()):
            await pc.close()
        self._pcs.clear()
        self._input_data.clear()
