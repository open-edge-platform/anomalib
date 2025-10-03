# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .filesystem import FolderDispatcher
from .mqtt import MqttDispatcher
from .webhook import WebhookDispatcher

Dispatcher = FolderDispatcher | MqttDispatcher | WebhookDispatcher

__all__ = ["Dispatcher", "FolderDispatcher", "MqttDispatcher", "WebhookDispatcher"]
