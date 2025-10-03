# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""This module contains the WebhookDispatcher class for dispatching images and predictions to a webhook endpoint."""

import logging
from typing import Any

import numpy as np
import requests
from anomalib.data import NumpyImageBatch as PredictionResult
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from pydantic_models.sink import WebhookSinkConfig
from services.dispatchers.base import BaseDispatcher

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_FACTOR = 0.3
RETRY_ON_STATUS = [500, 502, 503, 504]


class WebhookDispatcher(BaseDispatcher):
    def __init__(self, output_config: WebhookSinkConfig) -> None:
        """
        Initialize the WebhookDispatcher.
        Args:
            output_config: Configuration for the webhook-based output destination
        """
        super().__init__(output_config=output_config)
        self.webhook_url = output_config.webhook_url
        self.http_method = output_config.http_method
        self.headers = output_config.headers
        self.timeout = output_config.timeout
        self.session = requests.Session()
        retries = Retry(
            total=MAX_RETRIES,
            backoff_factor=BACKOFF_FACTOR,
            status_forcelist=RETRY_ON_STATUS,
            allowed_methods=["PATCH", "POST", "PUT"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def __send_to_webhook(self, payload: dict[str, Any]) -> None:
        logger.debug("Sending payload to webhook at %s", self.webhook_url)
        response = self.session.request(
            self.http_method, self.webhook_url, headers=self.headers, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        logger.debug("Response from webhook: %s", response.text)

    def _dispatch(
        self,
        original_image: np.ndarray,
        image_with_visualization: np.ndarray,
        predictions: PredictionResult,
    ) -> None:
        payload = self._create_payload(original_image, image_with_visualization, predictions)

        self.__send_to_webhook(payload)
