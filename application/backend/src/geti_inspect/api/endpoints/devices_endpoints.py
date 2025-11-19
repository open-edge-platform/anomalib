# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter

from geti_inspect.api.endpoints import API_PREFIX
from geti_inspect.pydantic_models.devices import DeviceList
from geti_inspect.utils.devices import Devices

device_router = APIRouter(
    prefix=API_PREFIX,
    tags=["Job"],
)


@device_router.get("/inference-devices")
async def get_inference_devices() -> DeviceList:
    """Endpoint to get list of supported devices for inference"""
    return DeviceList(devices=Devices.inference_devices())


@device_router.get("/training-devices")
async def get_training_devices() -> DeviceList:
    """Endpoint to get list of supported devices for training"""
    return DeviceList(devices=Devices.training_devices())
