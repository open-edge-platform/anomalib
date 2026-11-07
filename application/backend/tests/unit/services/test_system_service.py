# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from openvino.properties.device import Type as OVDeviceType

from services.system_service import SystemService


class TestSystemService:
    """Test cases for SystemService"""

    @pytest.fixture
    def fxt_system_service(self) -> SystemService:
        return SystemService()

    def test_get_memory_usage(self, fxt_system_service: SystemService):
        """Test getting memory usage"""
        used, total = fxt_system_service.get_memory_usage()

        assert used > 0
        assert total > 0
        assert used <= total

    def test_get_cpu_usage(self, fxt_system_service: SystemService):
        """Test getting CPU usage"""
        cpu_usage = fxt_system_service.get_cpu_usage()

        assert cpu_usage >= 0.0

    def test_get_devices_cpu_only(self, fxt_system_service: SystemService):
        """Test getting devices when only CPU is available"""
        with patch("services.system_service.torch") as mock_torch:
            # Simulate torch not being available
            mock_torch.xpu.is_available.return_value = False
            mock_torch.cuda.is_available.return_value = False
            mock_torch.mps.is_available.return_value = False

            devices = fxt_system_service.get_training_devices()

            assert len(devices) == 1
            assert devices[0].name == "CPU"
            assert devices[0].memory is None
            assert devices[0].index is None

    def test_get_devices_with_xpu(self, fxt_system_service: SystemService):
        """Test getting devices when Intel XPU is available"""
        with patch("services.system_service.torch") as mock_torch:
            # Mock XPU device
            mock_dp = MagicMock()
            mock_dp.name = "Intel(R) Graphics [0x7d41]"
            mock_dp.total_memory = 36022263808

            mock_torch.xpu.is_available.return_value = True
            mock_torch.xpu.device_count.return_value = 1
            mock_torch.xpu.get_device_properties.return_value = mock_dp

            # CUDA/MPS not available
            mock_torch.cuda.is_available.return_value = False
            mock_torch.mps.is_available.return_value = False

            devices = fxt_system_service.get_training_devices()

            assert len(devices) == 2
            assert devices[1].name == "Intel(R) Graphics [0x7d41]"
            assert devices[1].memory == 36022263808
            assert devices[1].index == 0

    def test_get_devices_with_cuda(self, fxt_system_service: SystemService):
        """Test getting devices when NVIDIA CUDA is available"""
        with patch("services.system_service.torch") as mock_torch:
            # XPU/MPS not available
            mock_torch.xpu.is_available.return_value = False
            mock_torch.mps.is_available.return_value = False

            # Mock CUDA device
            mock_dp = MagicMock()
            mock_dp.name = "NVIDIA GeForce RTX 4090"
            mock_dp.total_memory = 25769803776

            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 1
            mock_torch.cuda.get_device_properties.return_value = mock_dp

            devices = fxt_system_service.get_training_devices()

            assert len(devices) == 2
            assert devices[1].name == "NVIDIA GeForce RTX 4090"
            assert devices[1].memory == 25769803776
            assert devices[1].index == 0

    def test_get_devices_with_multiple_devices(self, fxt_system_service: SystemService):
        """Test getting devices when multiple GPUs are available"""
        with patch("services.system_service.torch") as mock_torch:
            # Mock XPU device
            mock_xpu_dp = MagicMock()
            mock_xpu_dp.name = "Intel(R) Graphics [0x7d41]"
            mock_xpu_dp.total_memory = 36022263808

            mock_torch.xpu.is_available.return_value = True
            mock_torch.xpu.device_count.return_value = 1
            mock_torch.xpu.get_device_properties.return_value = mock_xpu_dp

            # Mock CUDA device
            mock_cuda_dp = MagicMock()
            mock_cuda_dp.name = "NVIDIA GeForce RTX 4090"
            mock_cuda_dp.total_memory = 25769803776

            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 1
            mock_torch.cuda.get_device_properties.return_value = mock_cuda_dp

            mock_torch.mps.is_available.return_value = True

            devices = fxt_system_service.get_training_devices()

            assert len(devices) == 4

    def test_validate_device_cpu_always_valid(self, fxt_system_service: SystemService):
        """Test that CPU device is always valid"""
        assert fxt_system_service.validate_device("cpu") is True

    def test_validate_device_xpu_available(self, fxt_system_service: SystemService):
        """Test validating XPU device when available"""
        mock_xpu_dp = MagicMock()
        mock_xpu_dp.name = "Intel XPU"
        mock_xpu_dp.total_memory = 36022263808

        with patch("services.system_service.torch") as mock_torch:
            mock_torch.xpu.is_available.return_value = True
            mock_torch.cuda.is_available.return_value = False
            mock_torch.xpu.device_count.return_value = 2
            mock_torch.xpu.get_device_properties.return_value = mock_xpu_dp

            assert fxt_system_service.validate_device("xpu") is True
            assert fxt_system_service.validate_device("xpu-0") is True
            assert fxt_system_service.validate_device("xpu-1") is True
            assert fxt_system_service.validate_device("xpu-2") is False

    def test_validate_device_xpu_not_available(self, fxt_system_service: SystemService):
        """Test validating XPU device when not available"""
        with patch("services.system_service.torch") as mock_torch:
            mock_torch.xpu.is_available.return_value = False
            mock_torch.cuda.is_available.return_value = False

            assert fxt_system_service.validate_device("xpu") is False
            assert fxt_system_service.validate_device("xpu-0") is False

    def test_validate_device_cuda_available(self, fxt_system_service: SystemService):
        """Test validating CUDA device when available"""
        mock_cuda_dp = MagicMock()
        mock_cuda_dp.name = "NVIDIA GPU"
        mock_cuda_dp.total_memory = 25769803776

        with patch("services.system_service.torch") as mock_torch:
            mock_torch.xpu.is_available.return_value = False
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.device_count.return_value = 3
            mock_torch.cuda.get_device_properties.return_value = mock_cuda_dp

            assert fxt_system_service.validate_device("cuda") is True
            assert fxt_system_service.validate_device("cuda-0") is True
            assert fxt_system_service.validate_device("cuda-1") is True
            assert fxt_system_service.validate_device("cuda-2") is True
            assert fxt_system_service.validate_device("cuda-3") is False

    def test_validate_device_cuda_not_available(self, fxt_system_service: SystemService):
        """Test validating CUDA device when not available"""
        with patch("services.system_service.torch") as mock_torch:
            mock_torch.xpu.is_available.return_value = False
            mock_torch.cuda.is_available.return_value = False

            assert fxt_system_service.validate_device("cuda") is False
            assert fxt_system_service.validate_device("cuda-0") is False

    def test_get_inference_devices_with_multiple_devices(self, fxt_system_service: SystemService):
        """Test getting inference devices when multiple GPUs are available"""
        with patch("services.system_service.ov.Core") as mock_core_cls:
            mock_core = MagicMock()
            mock_core_cls.return_value = mock_core

            # Mock available devices: NPU and discrete Intel GPU
            mock_core.available_devices = ["NPU", "GPU.0"]

            def get_property_side_effect(device, prop):
                if prop == "FULL_DEVICE_NAME":
                    if device == "NPU":
                        return "Intel(R) AI Boost"
                    return "Intel(R) Graphics [0x7d41]"
                if prop == "DEVICE_TYPE":
                    return OVDeviceType.DISCRETE
                if prop == "GPU_DEVICE_TOTAL_MEM_SIZE":
                    return 36022263808
                if prop == "DEVICE_ID":
                    return 0
                return None

            mock_core.get_property.side_effect = get_property_side_effect

            inference_devices = fxt_system_service.get_inference_devices()

            # Should have CPU + NPU + XPU (Intel GPU)
            assert len(inference_devices) == 3
            assert inference_devices[0].type == "cpu"
            assert inference_devices[1].type == "npu"
            assert inference_devices[2].type == "xpu"
            assert not any(device.type == "cuda" for device in inference_devices)

    def test_validate_device_invalid_type(self, fxt_system_service: SystemService):
        """Test validating invalid device types"""
        with patch("services.system_service.torch") as mock_torch, pytest.raises(ValueError):
            mock_torch.xpu.is_available.return_value = False
            mock_torch.cuda.is_available.return_value = False

            assert fxt_system_service.validate_device("cpu-cpu") is False
            assert fxt_system_service.validate_device("cpu--1") is False
            assert fxt_system_service.validate_device("cpu-") is False
            assert fxt_system_service.validate_device("cpu-0.9") is False
            assert fxt_system_service.validate_device("1") is False
            assert fxt_system_service.validate_device("-1") is False
            assert fxt_system_service.validate_device("gpu") is False
            assert fxt_system_service.validate_device("tpu") is False
            assert fxt_system_service.validate_device("invalid") is False

    def test_get_device_info(self, fxt_system_service: SystemService):
        """Test getting device info"""
        with patch("services.system_service.torch") as mock_torch:
            # Mock XPU device
            mock_xpu_dp = MagicMock()
            mock_xpu_dp.name = "Intel(R) Graphics [0x7d41]"
            mock_xpu_dp.total_memory = 36022263808

            mock_torch.xpu.is_available.return_value = True
            mock_torch.xpu.device_count.return_value = 1
            mock_torch.xpu.get_device_properties.return_value = mock_xpu_dp

            # CUDA not available
            mock_torch.cuda.is_available.return_value = False

            device_info = fxt_system_service.get_device_info("cpu")

            assert device_info.type == "cpu"
            assert device_info.name == "CPU"
            assert device_info.memory is None
            assert device_info.index is None

            device_info = fxt_system_service.get_device_info("xpu-0")

            assert device_info.type == "xpu"
            assert device_info.name == "Intel(R) Graphics [0x7d41]"
            assert device_info.memory == 36022263808
            assert device_info.index == 0

    def test_get_device_info_invalid(self, fxt_system_service: SystemService):
        """Test getting device info for invalid device"""
        with pytest.raises(ValueError):
            fxt_system_service.get_device_info("xpu-999")

    def test_get_camera_devices(self, fxt_system_service: SystemService):
        """Test getting camera devices"""
        with patch("services.system_service.enumerate_cameras") as mock_enumerate_cameras:
            # Mock camera device
            mock_camera = MagicMock()
            mock_camera.name = "Integrated Camera"
            mock_camera.index = 1400

            mock_enumerate_cameras.return_value = [mock_camera]

            camera_devices = fxt_system_service.get_camera_devices()

            assert len(camera_devices) == 1
            assert camera_devices[0].name == "Integrated Camera [1400]"
            assert camera_devices[0].index == 1400
