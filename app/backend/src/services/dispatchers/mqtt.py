# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import threading
import time
from typing import Any

import numpy as np
from anomalib.data import NumpyImageBatch as PredictionResult

from pydantic_models.sink import MqttSinkConfig
from services.dispatchers.base import BaseDispatcher

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)
MAX_RETRIES = 3
RETRY_DELAY = 1
CONNECT_TIMEOUT = 10


class MqttDispatcher(BaseDispatcher):
    def __init__(
        self,
        output_config: MqttSinkConfig,
        mqtt_client: "mqtt.Client | None" = None,
        track_messages: bool | None = False,
    ) -> None:
        """
        Initialize the MqttDispatcher.

        Args:
            output_config: Configuration for the MQTT destination
            mqtt_client: MQTT client
            track_messages: Flag to track MQTT messages (useful for debugging/testing)

        Raises:
            ImportError: If paho-mqtt is not installed
            ConnectionError: If unable to connect to MQTT broker
        """
        if mqtt is None:
            raise ImportError("paho-mqtt is required for MQTT dispatcher.")

        super().__init__(output_config)
        self.broker_host = output_config.broker_host
        self.broker_port = output_config.broker_port
        self.topic = output_config.topic
        self.username, self.password = output_config.get_credentials()

        self._connected = False
        self._connection_lock = threading.Lock()
        self._connection_event = threading.Event()
        self._track_messages = track_messages
        self._published_messages: list[dict] = []

        self.client = mqtt_client or self._create_default_client()
        self._connect()

    def _create_default_client(self) -> "mqtt.Client":
        client_id = f"dispatcher_{int(time.time())}"
        client = mqtt.Client(client_id=client_id)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        if self.username is not None and self.password is not None:
            client.username_pw_set(self.username, self.password)
        return client

    def _connect(self) -> None:
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(
                    "Connecting to MQTT broker at %s:%s (attempt %s)", self.broker_host, self.broker_port, attempt + 1
                )
                self.client.connect(self.broker_host, self.broker_port)
                self.client.loop_start()
                if self._connection_event.wait(CONNECT_TIMEOUT):
                    return
                logger.warning("Connection timeout after %s seconds", CONNECT_TIMEOUT)
            except Exception as e:
                logger.exception("Connection failed %s", e)
                time.sleep(RETRY_DELAY * (attempt + 1))
        raise ConnectionError("Failed to connect to MQTT broker")

    def _on_connect(self, _client: "mqtt.Client", _userdata: Any, _flags: dict[str, int], rc: int):
        if rc == 0:
            self._connected = True
            self._connection_event.set()
            logger.info("Connected to MQTT broker")
        else:
            logger.error("MQTT connect failed with code %s", rc)

    def _on_disconnect(self, _client: "mqtt.Client", _userdata: Any, rc: int):
        self._connected = False
        self._connection_event.clear()
        logger.warning("MQTT disconnected (rc=%s)", rc)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def __publish_message(self, topic: str, payload: dict[str, Any]) -> None:
        if not self._connected:
            logger.warning("Client not connected. Reconnecting...")
            try:
                self._connect()
            except ConnectionError:
                logger.exception("Reconnect failed")

        try:
            PredictionResult = self.client.publish(topic, json.dumps(payload))
            if PredictionResult.rc == mqtt.MQTT_ERR_SUCCESS and self._track_messages:
                self._published_messages.append({"topic": topic, "payload": payload})
            logger.error(f"Publish failed: {mqtt.error_string(PredictionResult.rc)}")
        except ValueError:
            logger.exception("Invalid payload for MQTT publish")

    def _dispatch(
        self,
        original_image: np.ndarray,
        image_with_visualization: np.ndarray,
        predictions: PredictionResult,
    ) -> None:
        payload = self._create_payload(original_image, image_with_visualization, predictions)

        self.__publish_message(self.topic, payload)

    def get_published_messages(self) -> list:
        return self._published_messages.copy()

    def clear_published_messages(self) -> None:
        self._published_messages.clear()

    def close(self) -> None:
        err = self.client.loop_stop()
        if err != mqtt.MQTT_ERR_SUCCESS:
            logger.warning(f"Error stopping MQTT loop: {mqtt.error_string(err)}")
        err = self.client.disconnect()
        if err != mqtt.MQTT_ERR_SUCCESS:
            logger.warning(f"Error disconnecting MQTT client: {mqtt.error_string(err)}")
        self._connected = False
        self._connection_event.clear()
