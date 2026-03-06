# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np

JPEG_QUALITY = 85


class JPEGEncoder:
    def __init__(self, quality: int = JPEG_QUALITY):
        self.quality = quality

    def encode(self, frame: np.ndarray) -> bytes | None:
        """Encode an RGB frame to JPEG bytes.

        Args:
            frame (np.ndarray): RGB frame to encode.

        Returns:
            bytes | None: Encoded JPEG bytes, or None if encoding fails.
        """
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        success, jpeg = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not success:
            return None
        return jpeg.tobytes()
