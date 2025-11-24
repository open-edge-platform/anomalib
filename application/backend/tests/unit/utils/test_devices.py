# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import pytest

from utils.devices import Devices


class TestDevices:
    """Test suite for Devices utility class."""

    @patch("utils.devices.sys.platform", "darwin")
    @patch("utils.devices.Devices._get_webcam_devices_worker_process")
    def test_get_webcam_devices_macos(self, mock_worker: MagicMock) -> None:
        """Test get_webcam_devices on macOS calls the worker process."""
        expected_cameras = [{"index": 0, "name": "Test Camera"}]
        mock_worker.return_value = expected_cameras

        cameras = Devices.get_webcam_devices()

        assert cameras == expected_cameras
        mock_worker.assert_called_once()

    @patch("utils.devices.sys.platform", "linux")
    @patch("utils.devices.cv2_enumerate_cameras.enumerate_cameras")
    @patch("utils.devices.Devices._get_webcam_devices_worker_process")
    def test_get_webcam_devices_non_macos(self, mock_worker: MagicMock, mock_enumerate: MagicMock) -> None:
        """Test get_webcam_devices on non-macOS does NOT call worker process."""
        # Setup mock return value
        mock_cam = MagicMock()
        mock_cam.index = 1
        mock_cam.name = "Linux Camera"
        mock_enumerate.return_value = [mock_cam]

        cameras = Devices.get_webcam_devices()

        assert len(cameras) == 1
        assert cameras[0]["index"] == 1
        assert cameras[0]["name"] == "Linux Camera"
        mock_worker.assert_not_called()
        mock_enumerate.assert_called_once()

    @patch("utils.devices.multiprocessing.get_context")
    @patch("utils.devices._enumerate_cameras_worker_pool")
    def test_get_webcam_devices_worker_process_success(
        self, mock_worker_func: MagicMock, mock_get_context: MagicMock
    ) -> None:
        """Test _get_webcam_devices_worker_process handles successful execution."""
        # Setup mocks
        mock_pool = MagicMock()
        mock_context = MagicMock()
        mock_context.Pool.return_value.__enter__.return_value = mock_pool
        mock_get_context.return_value = mock_context

        expected_cameras = [{"index": 0, "name": "Worker Camera"}]
        mock_pool.apply.return_value = expected_cameras

        result = Devices._get_webcam_devices_worker_process()

        assert result == expected_cameras
        mock_get_context.assert_called_with("spawn")
        # Verify it was called with the mocked worker function
        mock_pool.apply.assert_called_once_with(mock_worker_func)

    @patch("utils.devices.multiprocessing.get_context")
    def test_get_webcam_devices_worker_process_failure(self, mock_get_context: MagicMock) -> None:
        """Test _get_webcam_devices_worker_process handles exceptions."""
        # Setup mocks
        mock_pool = MagicMock()
        mock_context = MagicMock()
        mock_context.Pool.return_value.__enter__.return_value = mock_pool
        mock_get_context.return_value = mock_context

        mock_pool.apply.side_effect = Exception("Worker error")

        with pytest.raises(RuntimeError, match="Camera enumeration failed"):
            Devices._get_webcam_devices_worker_process()
