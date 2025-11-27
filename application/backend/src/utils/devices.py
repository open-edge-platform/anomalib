# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import sys
from functools import lru_cache
from typing import TypedDict

import cv2_enumerate_cameras
import openvino as ov
from lightning.pytorch.accelerators import AcceleratorRegistry


class CameraInfo(TypedDict):
    index: int
    name: str


class Devices:
    """Utility class for device-related operations."""

    @staticmethod
    def get_webcam_devices() -> list[CameraInfo]:
        """Get list of available webcam devices.

        Returns:
            list[CameraInfo]: List of dictionaries containing camera index and name.
        """
        # TODO: remove branching after: https://github.com/lukehugh/cv2_enumerate_cameras/pull/13
        if sys.platform == "darwin":
            import AVFoundation  # noqa: PLC0415 # late import to avoid issues on non-macOS systems

            # macOS hardware connection notifications are delivered asynchronously via the NSRunLoop.
            # Running it once is enough to process pending notifications and update the list.
            run_loop = AVFoundation.NSRunLoop.currentRunLoop()
            run_loop.runUntilDate_(AVFoundation.NSDate.dateWithTimeIntervalSinceNow_(0))  # runs loop once

        return [{"index": cam.index, "name": cam.name} for cam in cv2_enumerate_cameras.enumerate_cameras()]

    @staticmethod
    @lru_cache
    def training_devices() -> list[str]:
        """Get list of supported devices for training."""
        devices = []
        for device_name, device_info in AcceleratorRegistry.items():
            accelerator = device_info["accelerator"]
            if accelerator.is_available():
                devices.append(device_name.casefold())
        return devices

    @staticmethod
    @lru_cache
    def inference_devices() -> list[str]:
        """Get list of supported devices for inference."""
        ov_core = ov.Core()
        return [device.casefold() for device in ov_core.available_devices]

    @classmethod
    @lru_cache
    def _is_device_supported(cls, device_name: str, for_training: bool = False) -> bool:
        """Check if a device is supported for inference or training.

        Args:
            device_name (str): Name of the device to check.
            for_training (bool): If True, check for training devices; otherwise, check for inference devices.

        Returns:
            bool: True if the device is supported, False otherwise.
        """
        device_name = device_name.casefold()
        if for_training:
            return device_name in cls.training_devices()
        return device_name in cls.inference_devices()

    @classmethod
    def is_device_supported_for_inference(cls, device_name: str) -> bool:
        """Check if a device is supported for inference.

        Args:
            device_name (str): Name of the device to check.

        Returns:
            bool: True if the device is supported for inference, False otherwise.
        """
        return cls._is_device_supported(device_name, for_training=False)

    @classmethod
    def is_device_supported_for_training(cls, device_name: str) -> bool:
        """Check if a device is supported for training.

        Args:
            device_name (str): Name of the device to check.

        Returns:
            bool: True if the device is supported for training, False otherwise.
        """
        return cls._is_device_supported(device_name, for_training=True)
