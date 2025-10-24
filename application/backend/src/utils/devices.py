# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from functools import lru_cache

import openvino as ov
from lightning.pytorch.accelerators import AcceleratorRegistry


@lru_cache
def inference_devices() -> list[str]:
    """Get list of supported devices for inference."""
    ov_core = ov.Core()
    return ov_core.available_devices


@lru_cache
def training_devices() -> list[str]:
    """Get list of supported devices for training."""
    devices = []
    for device_name, device_info in AcceleratorRegistry.items():
        accelerator = device_info["accelerator"]
        if accelerator.is_available():
            devices.append(device_name)
    return devices
