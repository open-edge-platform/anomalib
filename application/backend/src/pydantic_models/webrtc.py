# Copyright (C) 2025 Intel Corporation
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
