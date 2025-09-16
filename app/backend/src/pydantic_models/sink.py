# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from os import getenv
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from pydantic_models.base import BaseIDNameModel

MQTT_USERNAME = "MQTT_USERNAME"
MQTT_PASSWORD = "MQTT_PASSWORD"  # noqa: S105


class SinkType(StrEnum):
    DISCONNECTED = "disconnected"
    FOLDER = "folder"
    MQTT = "mqtt"
    ROS = "ros"
    WEBHOOK = "webhook"


class OutputFormat(StrEnum):
    IMAGE_ORIGINAL = "image_original"
    IMAGE_WITH_PREDICTIONS = "image_with_predictions"
    PREDICTIONS = "predictions"


class BaseSinkConfig(BaseIDNameModel):
    output_formats: list[OutputFormat]
    rate_limit: float | None = None  # Rate limit in Hz, None means no limit


class DisconnectedSinkConfig(BaseSinkConfig):
    sink_type: Literal[SinkType.DISCONNECTED] = SinkType.DISCONNECTED
    name: str = "No Sink"
    output_formats: list[OutputFormat] = []

    model_config = {
        "json_schema_extra": {
            "example": {
                "sink_type": "disconnected",
                "name": "No Sink",
                "id": "00000000-0000-0000-0000-000000000000",
            }
        }
    }


class FolderSinkConfig(BaseSinkConfig):
    sink_type: Literal[SinkType.FOLDER]
    folder_path: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "b5787c06-964b-4097-8eca-238b8cf79fc8",
                "sink_type": "folder",
                "name": "Local Folder",
                "folder_path": "/path/to/output",
                "output_formats": ["image_original", "image_with_predictions", "predictions"],
                "rate_limit": 0.2,
            }
        }
    }


class MqttSinkConfig(BaseSinkConfig):
    sink_type: Literal[SinkType.MQTT]
    broker_host: str
    broker_port: int
    topic: str
    auth_required: bool = False

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "c1a70159-9c9e-4f02-821a-02576321056c",
                "sink_type": "mqtt",
                "name": "Local MQTT Broker",
                "broker_host": "localhost",
                "broker_port": 1883,
                "topic": "predictions",
                "output_formats": ["predictions"],
                "auth_required": True,
            }
        }
    }

    def get_credentials(self) -> tuple[str | None, str | None]:
        """Configure stream URL with authentication if required."""
        if not self.auth_required:
            return None, None

        username = getenv(MQTT_USERNAME)
        password = getenv(MQTT_PASSWORD)

        if not username or not password:
            raise RuntimeError("MQTT credentials not provided.")

        return username, password


class RosSinkConfig(BaseSinkConfig):
    sink_type: Literal[SinkType.ROS]
    topic: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "6f1d96ac-db38-42a9-9a11-142d404f493f",
                "sink_type": "ros",
                "name": "ROS2 Predictions Topic",
                "topic": "/predictions",
                "output_formats": ["predictions"],
            }
        }
    }


HttpMethod = Literal["POST", "PUT", "PATCH"]
HttpHeaders = dict[str, str]


class WebhookSinkConfig(BaseSinkConfig):
    sink_type: Literal[SinkType.WEBHOOK]
    webhook_url: str
    http_method: HttpMethod = "POST"
    headers: HttpHeaders | None = None
    timeout: int = 10  # seconds

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "39ba53e5-9a03-44fc-b78a-83245cf14676",
                "sink_type": "webhook",
                "name": "Webhook Endpoint",
                "webhook_url": "https://example.com/webhook",
                "http_method": "PUT",
                "headers": {"Authorization": "Bearer YOUR_TOKEN"},
                "output_formats": ["predictions"],
            }
        }
    }


Sink = Annotated[
    FolderSinkConfig | MqttSinkConfig | RosSinkConfig | WebhookSinkConfig | DisconnectedSinkConfig,
    Field(discriminator="sink_type"),
]

SinkAdapter: TypeAdapter[Sink] = TypeAdapter(Sink)
