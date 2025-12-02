# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

from utils.devices import Devices


class TestDevices:
    """Test suite for Devices utility class."""

    @patch("utils.devices.cv2_enumerate_cameras.enumerate_cameras")
    def test_get_webcam_devices(self, mock_enumerate: MagicMock) -> None:
        """Test get_webcam_devices."""
        # Setup mock return value
        mock_cam = MagicMock()
        mock_cam.index = 1
        mock_cam.name = "Linux Camera"
        mock_enumerate.return_value = [mock_cam]

        cameras = Devices.get_webcam_devices()

        assert len(cameras) == 1
        assert cameras[0]["index"] == 1
        assert cameras[0]["name"] == "Linux Camera"
        mock_enumerate.assert_called_once()
