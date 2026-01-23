# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import platform
import re
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import cv2
import psutil
import torch
from cv2_enumerate_cameras import enumerate_cameras

from pydantic_models.system import CameraInfo, DeviceInfo, DeviceType, LibraryVersions, SystemInfo
from settings import get_settings

DEVICE_PATTERN = re.compile(r"^(cpu|xpu|cuda)(-(\d+))?$")
DEFAULT_DEVICE = "cpu"
CV2_BACKENDS = {
    "Windows": cv2.CAP_MSMF,
    "Linux": cv2.CAP_V4L2,
    "Darwin": cv2.CAP_AVFOUNDATION,
}


class SystemService:
    """Service to get system information"""

    def __init__(self) -> None:
        self.process = psutil.Process()

    def get_memory_usage(self) -> tuple[float, float]:
        """
        Get the memory usage of the process

        Returns:
            tuple[float, float]: Used memory in MB and total available memory in MB
        """
        memory_info = psutil.virtual_memory()
        return self.process.memory_info().rss / (1024 * 1024), memory_info.total / (1024 * 1024)

    def get_cpu_usage(self) -> float:
        """
        Get the CPU usage of the process

        Returns:
            float: CPU usage in percentage
        """
        return self.process.cpu_percent(interval=None)

    @staticmethod
    def get_devices() -> list[DeviceInfo]:
        """
        Get available compute devices (CPU, GPUs, ...)

        Returns:
            list[DeviceInfo]: List of available devices
        """
        # CPU is always available
        devices: list[DeviceInfo] = [DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)]

        # Check for Intel XPU devices
        if torch.xpu.is_available():
            for device_idx in range(torch.xpu.device_count()):
                xpu_dp = torch.xpu.get_device_properties(device_idx)
                devices.append(
                    DeviceInfo(
                        type=DeviceType.XPU,
                        name=xpu_dp.name,
                        memory=xpu_dp.total_memory,
                        index=device_idx,
                    ),
                )

        # Check for NVIDIA CUDA devices
        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                cuda_dp = torch.cuda.get_device_properties(device_idx)
                devices.append(
                    DeviceInfo(
                        type=DeviceType.CUDA,
                        name=cuda_dp.name,
                        memory=cuda_dp.total_memory,
                        index=device_idx,
                    ),
                )

        # Check if Apple MPS is available
        if torch.mps.is_available():
            devices.append(DeviceInfo(type=DeviceType.MPS, name="MPS", memory=None, index=None))

        return devices

    @classmethod
    def get_inference_devices(cls) -> list[DeviceInfo]:
        """
        Get available compute devices for inference (CPU, XPU, ...)

        Returns:
            list[DeviceInfo]: List of available devices
        """
        return [device for device in cls.get_devices() if device.type not in {DeviceType.CUDA, DeviceType.MPS}]

    @classmethod
    def get_training_devices(cls) -> list[DeviceInfo]:
        """
        Get available compute devices for training (CPUs, XPUs, GPUs, ...)

        Returns:
            list[DeviceInfo]: List of available training devices
        """
        return cls.get_devices()  # currently same as get_devices, can be customized later with filters

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
            training_devices_names = [device.name.casefold() for device in cls.get_training_devices()]
            training_devices_types = [device.type.casefold() for device in cls.get_training_devices()]
            return device_name in training_devices_names or device_name in training_devices_types
        inference_devices_names = [device.name.casefold() for device in cls.get_inference_devices()]
        inference_devices_types = [device.type.casefold() for device in cls.get_inference_devices()]
        return device_name in inference_devices_names or device_name in inference_devices_types

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

    def validate_device(self, device_str: str) -> bool:
        """
        Validate if a device string is available on the system.

        Args:
            device_str: Device string in format '<target>[-<index>]' (e.g., 'cpu', 'xpu', 'cuda', 'xpu-2', 'cuda-1')

        Returns:
            bool: True if the device is available, False otherwise
        """
        device_type, device_index = self._parse_device(device_str)

        # CPU is always available
        if device_type == DeviceType.CPU:
            return True

        # Check if desired device is among available devices
        available_devices = self.get_devices()
        for available_device in available_devices:
            if device_type == available_device.type and device_index == (available_device.index or 0):
                return True

        return False

    def get_device_info(self, device_str: str) -> DeviceInfo:
        """
        Get DeviceInfo for a given device string.

        Args:
            device_str: Device string in format '<target>[-<index>]' (e.g., 'cpu', 'xpu', 'cuda', 'xpu-2', 'cuda-1')

        Returns:
            DeviceInfo: Information about the specified device
        """
        if not self.validate_device(device_str):
            raise ValueError(f"Device '{device_str}' is not available on the system.")

        device_type, device_index = self._parse_device(device_str)
        if device_type == DeviceType.CPU:
            return DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)
        return next(
            device for device in self.get_devices() if device.type == device_type and device.index == device_index
        )

    @staticmethod
    def _parse_device(device_str: str) -> tuple[DeviceType, int]:
        """
        Parse device string into type and index

        Args:
            device_str: Device string in format '<target>[-<index>]' (e.g., 'cpu', 'xpu', 'cuda', 'xpu-2', 'cuda-1')

        Returns:
            tuple[str, int]: Device type and index
        """
        m = DEVICE_PATTERN.match(device_str.lower())
        if not m:
            raise ValueError(f"Invalid device string: {device_str}")

        device_type, _, device_index = m.groups()
        device_index = int(device_index) if device_index is not None else 0
        return DeviceType(device_type.lower()), device_index

    @staticmethod
    def get_camera_devices() -> list[CameraInfo]:
        """
        Get available camera devices.
        Camera names are formatted as "<camera_name> [<index>]".

        Returns:
            list[CameraInfo]: List of available camera devices
        """
        if (backend := CV2_BACKENDS.get(platform.system())) is None:
            raise RuntimeError(f"Unsupported platform: {platform.system()}")

        return [CameraInfo(index=cam.index, name=f"{cam.name} [{cam.index}]") for cam in enumerate_cameras(backend)]

    @staticmethod
    def get_package_version(package_name: str) -> str | None:
        """Safely get version of an installed Python package.

        Args:
            package_name: Name of the package to check. If the package is not installed, return N/A.

        Returns:
            Version string if package is installed
        """
        try:
            return version(package_name)
        except PackageNotFoundError:
            return None

    @classmethod
    def get_library_versions(cls) -> LibraryVersions:
        """Collect versions of ML/DL libraries.

        Returns:
            LibraryVersions with version info for each library.
        """
        return LibraryVersions(
            anomalib=cls.get_package_version("anomalib"),
            python=platform.python_version(),
            openvino=cls.get_package_version("openvino"),
            pytorch=cls.get_package_version("torch"),
            lightning=cls.get_package_version("lightning"),
            torchmetrics=cls.get_package_version("torchmetrics"),
            onnx=cls.get_package_version("onnx"),
            cuda=torch.version.cuda if torch.cuda.is_available() else None,
            cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            xpu_driver=(
                getattr(torch.xpu.get_device_properties(torch.xpu.current_device()), "driver_version", "N/A")
                if hasattr(torch, "xpu") and torch.xpu.is_available()
                else None
            ),
        )

    def get_system_info(self) -> SystemInfo:
        """Get system information for feedback and diagnostics.

        Returns:
            SystemInfo containing OS details, app version, library versions,
            and devices info.
        """
        return SystemInfo(
            os_name=platform.system(),
            os_version=platform.release(),
            platform=platform.platform(),
            app_version=get_settings().version,
            libraries=self.get_library_versions(),
            devices=self.get_devices(),
        )
