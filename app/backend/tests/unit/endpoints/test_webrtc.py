# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from fastapi import status

from api.dependencies import get_webrtc_manager
from main import app
from pydantic_models.webrtc import Answer, InputData, Offer
from webrtc.manager import WebRTCManager


@pytest.fixture
def fxt_webrtc_manager():
    webrtc_manager = MagicMock(spec=WebRTCManager)
    app.dependency_overrides[get_webrtc_manager] = lambda: webrtc_manager
    return webrtc_manager


@pytest.fixture
def fxt_offer() -> Offer:
    return Offer(sdp="test_sdp", type="offer", webrtc_id="test_id")


@pytest.fixture
def fxt_answer() -> Answer:
    return Answer(sdp="test_sdp", type="answer")


@pytest.fixture
def fxt_input_data() -> InputData:
    return InputData(webrtc_id="test_id", conf_threshold=0.5)


class TestWebRTCEndpoints:
    def test_create_webrtc_offer_success(self, fxt_client, fxt_webrtc_manager, fxt_offer, fxt_answer):
        fxt_webrtc_manager.handle_offer.return_value = fxt_answer
        resp = fxt_client.post("/api/webrtc/offer", json=fxt_offer.model_dump(mode="json"))
        assert resp.status_code == status.HTTP_200_OK
        assert resp.json() == fxt_answer.model_dump()
        fxt_webrtc_manager.handle_offer.assert_called_once()

    def test_create_webrtc_offer_failure(self, fxt_client, fxt_webrtc_manager, fxt_offer):
        fxt_webrtc_manager.handle_offer.side_effect = Exception("fail")
        resp = fxt_client.post("/api/webrtc/offer", json=fxt_offer.model_dump(mode="json"))
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "fail" in resp.json()["detail"]
        fxt_webrtc_manager.handle_offer.assert_called_once()

    def test_create_webrtc_offer_invalid_payload(self, fxt_client):
        resp = fxt_client.post("/api/webrtc/offer", json={"sdp": 123})
        assert resp.status_code == status.HTTP_400_BAD_REQUEST

    def test_webrtc_input_hook_success(self, fxt_client, fxt_webrtc_manager, fxt_input_data):
        resp = fxt_client.post("/api/webrtc/input_hook", json=fxt_input_data.model_dump(mode="json"))
        assert resp.status_code == status.HTTP_200_OK
        fxt_webrtc_manager.set_input.assert_called_once()

    def test_webrtc_input_hook_invalid_payload(self, fxt_client, fxt_webrtc_manager):
        resp = fxt_client.post("/api/webrtc/input_hook", json={"wrong": "field"})
        assert resp.status_code == status.HTTP_400_BAD_REQUEST
