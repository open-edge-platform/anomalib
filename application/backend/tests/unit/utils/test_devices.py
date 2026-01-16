# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

from utils.devices import Devices


class TestDevices:
    """Test suite for Devices utility class."""

    @patch("utils.devices.enumerate_cameras")
    def test_get_webcam_devices(self, mock_enumerate: MagicMock) -> None:
        """Test get_webcam_devices."""
        # Setup mock return value
        mock_cam = MagicMock()
        mock_cam.index = 1
        mock_cam.name = "Linux Camera"
        mock_enumerate.return_value = [mock_cam]

        cameras = Devices.get_camera_devices()

        assert len(cameras) == 1
        assert cameras[0]["index"] == 1
        assert cameras[0]["name"] == "Linux Camera [1]"
        mock_enumerate.assert_called_once()

    @patch("utils.devices.enumerate_cameras")
    def test_get_webcam_devices_duplicate_names(self, mock_enumerate: MagicMock) -> None:
        """Ensure duplicate camera names are suffixed to remain unique."""

        def build_cam(index: int, name: str) -> MagicMock:
            cam = MagicMock()
            cam.index = index
            cam.name = name
            return cam

        mock_enumerate.return_value = [build_cam(0, "nikon"), build_cam(1, "nikon")]
        cameras = Devices.get_camera_devices()
        assert [cam["name"] for cam in cameras] == ["nikon [0]", "nikon [1]"]
        assert [cam["index"] for cam in cameras] == [0, 1]
