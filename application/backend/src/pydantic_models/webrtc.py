# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field


class InputData(BaseModel):
    webrtc_id: str
    conf_threshold: float = Field(ge=0, le=1)


class Offer(BaseModel):
    webrtc_id: str
    sdp: str
    type: str


class Answer(BaseModel):
    sdp: str
    type: str


class WebRTCIceServer(BaseModel):
    """ICE server configuration for WebRTC connections."""

    urls: str = Field(..., description="ICE server URL (STUN/TURN)")
    username: str | None = Field(None, description="Username for TURN server authentication")
    credential: str | None = Field(None, description="Credential for TURN server authentication")

    model_config = {
        "json_schema_extra": {
            "example": {"urls": "turn:192.168.1.100:3478?transport=tcp", "username": "user", "credential": "password"},
        },
    }


class WebRTCConfigResponse(BaseModel):
    """WebRTC configuration response."""

    ice_servers: list[WebRTCIceServer] = Field(
        ...,
        alias="iceServers",
        description="List of ICE servers for WebRTC connection establishment",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "iceServers": [
                    {"urls": "turn:192.168.1.100:3478?transport=tcp", "username": "user", "credential": "password"},
                ],
            },
        },
    }
