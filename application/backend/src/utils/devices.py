# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import platform
from functools import lru_cache
from typing import TypedDict

import cv2
import openvino as ov
from cv2_enumerate_cameras import enumerate_cameras
from lightning.pytorch.accelerators import AcceleratorRegistry

CV2_BACKENDS = {
    "Windows": cv2.CAP_MSMF,
    "Linux": cv2.CAP_V4L2,
    "Darwin": cv2.CAP_AVFOUNDATION,
}


class CameraInfo(TypedDict):
    index: int
    name: str


class Devices:
    """Utility class for device-related operations."""

    @staticmethod
    def get_camera_devices() -> list[CameraInfo]:
        """
        Get list of available camera devices.
        Camera names are formatted as "<camera_name> [<index>]".

        Returns:
            list[CameraInfo]: List of dictionaries containing camera index and name.
        """
        if (backend := CV2_BACKENDS.get(platform.system())) is None:
            raise RuntimeError(f"Unsupported platform: {platform.system()}")

        return [CameraInfo(index=cam.index, name=f"{cam.name} [{cam.index}]") for cam in enumerate_cameras(backend)]

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
