# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time

import cv2
import numpy as np
from loguru import logger

from entities.base_opencv_stream import BaseOpenCVStream
from pydantic_models.source import IPCameraSourceConfig, SourceType


class IPCameraStream(BaseOpenCVStream):
    """Video stream implementation using IP camera via OpenCV."""

    MAX_RETRIES = 3
    BACKOFF_FACTOR = 0.1

    def __init__(self, config: IPCameraSourceConfig) -> None:
        """Initialize IP camera stream."""
        super().__init__(
            source=config.get_configured_stream_url(),
            source_type=SourceType.IP_CAMERA,
            stream_url=config.stream_url,  # Original stream URL is kept for metadata
        )
        logger.info("IP camera stream initialized")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _handle_read_failure(self) -> np.ndarray:
        """Handle IP camera read failure"""
        if self.cap is None:
            raise RuntimeError("Video capture not initialized")

        for attempt in range(self.MAX_RETRIES):
            logger.warning(f"Attempt {attempt + 1}: Failed to capture frame from IP camera, retrying...")
            self.release()
            self._initialize_capture()
            ret, frame = self.cap.read()
            if ret:
                logger.info("Successfully reconnected to IP camera stream.")
                return frame
            time.sleep(self.BACKOFF_FACTOR * (attempt + 1))  # Exponential backoff

        raise RuntimeError("Failed to capture frame from IP camera after multiple retries")

    def is_real_time(self) -> bool:
        return True

    def __enter__(self) -> "IPCameraStream":
        return self
