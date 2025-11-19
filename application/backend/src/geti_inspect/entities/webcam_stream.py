# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from geti_inspect.entities.base_opencv_stream import BaseOpenCVStream
from geti_inspect.pydantic_models import SourceType


class WebcamStream(BaseOpenCVStream):
    """Video stream implementation using webcam via OpenCV."""

    def __init__(self, device_id: int = 0) -> None:
        """Initialize webcam stream."""
        super().__init__(source=device_id, source_type=SourceType.WEBCAM, device_id=device_id)

    def is_real_time(self) -> bool:
        return True
