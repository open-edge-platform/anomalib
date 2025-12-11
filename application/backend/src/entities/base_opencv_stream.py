# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from abc import ABC
from typing import Any

import cv2
import numpy as np

from entities.stream_data import StreamData
from entities.video_stream import VideoStream
from pydantic_models import SourceType


class BaseOpenCVStream(VideoStream, ABC):
    """Base class for OpenCV-based video streams with common functionality."""

    def __init__(self, source: str | int, source_type: SourceType, **metadata) -> None:
        """Initialize OpenCV stream.

        Args:
            source: Video source (device ID, file path, or URL)
            source_type: Type of the video source
            **metadata: Additional metadata for the stream
        """
        self.source = source
        self.source_type = source_type
        self.metadata = metadata
        self.cap: cv2.VideoCapture
        self._initialize_capture()

    def _initialize_capture(self) -> None:
        """Initialize the OpenCV VideoCapture."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

    def _read_frame(self) -> np.ndarray:
        """Read a frame from the capture device."""
        if self.cap is None:
            raise RuntimeError("Video capture not initialized")

        ret, frame = self.cap.read()
        if not ret:
            return self._handle_read_failure()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _handle_read_failure(self) -> np.ndarray:
        """Handle frame read failure. Override in subclasses for specific behavior."""
        raise RuntimeError(f"Failed to capture frame from {self.source_type.value}")

    def _get_source_metadata(self) -> dict[str, Any]:
        """Get metadata specific to this source."""
        return {"source_type": self.source_type.value, **self.metadata}

    def get_data(self) -> StreamData:
        """Get the latest frame from the video stream."""
        frame = self._read_frame()
        return StreamData(
            frame_data=frame,
            timestamp=time.time(),
            source_metadata=self._get_source_metadata(),
        )

    def release(self) -> None:
        """Release OpenCV VideoCapture resources."""
        if self.cap is not None:
            self.cap.release()
