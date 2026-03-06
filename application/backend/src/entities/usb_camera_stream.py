# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from entities.base_opencv_stream import BaseOpenCVStream
from pydantic_models import SourceType


class UsbCameraStream(BaseOpenCVStream):
    """Video stream implementation using USB camera via OpenCV."""

    def __init__(self, device_id: int = 0) -> None:
        """Initialize USB camera stream."""
        super().__init__(source=device_id, source_type=SourceType.USB_CAMERA, device_id=device_id)

    def is_real_time(self) -> bool:
        return True
