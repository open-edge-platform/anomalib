# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import multiprocessing
import sys
from functools import lru_cache
from typing import TypedDict

import cv2_enumerate_cameras
import openvino as ov
from lightning.pytorch.accelerators import AcceleratorRegistry
from loguru import logger


class CameraInfo(TypedDict):
    index: int
    name: str


def _enumerate_cameras_worker_pool() -> list[CameraInfo]:
    """Worker function to enumerate cameras (for use with multiprocessing.Pool)."""
    return [
        {"index": camera_info.index, "name": camera_info.name}
        for camera_info in cv2_enumerate_cameras.enumerate_cameras()
    ]


class Devices:
    """Utility class for device-related operations."""

    @staticmethod
    def get_webcam_devices() -> list[CameraInfo]:
        """Get list of available webcam devices.

        On macOS, uses a separate process to enumerate cameras due to AVFoundation caching issues.
        May raise RuntimeError if camera enumeration fails (especially on macOS).

        Returns:
            list[CameraInfo]: List of dictionaries containing camera index and name.
        """
        if sys.platform == "darwin":
            # On macOS, use a separate process because AVFoundation uses caching that prevents device list update
            return Devices._get_webcam_devices_worker_process()

        return [{"index": cam.index, "name": cam.name} for cam in cv2_enumerate_cameras.enumerate_cameras()]

    @staticmethod
    def _get_webcam_devices_worker_process() -> list[CameraInfo]:
        """Get webcam devices using a separate process."""
        try:
            with multiprocessing.get_context("spawn").Pool(processes=1) as pool:
                return pool.apply(_enumerate_cameras_worker_pool)
        except Exception as e:
            logger.error(f"Camera enumeration failed: {e}")
            raise RuntimeError(f"Camera enumeration failed: {e}") from e

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
