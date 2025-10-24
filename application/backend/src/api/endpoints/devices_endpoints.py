# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter

from api.endpoints import API_PREFIX
from pydantic_models.devices import DeviceList
from utils.devices import inference_devices, training_devices

device_router = APIRouter(
    prefix=API_PREFIX,
    tags=["Job"],
)


@device_router.get("/inference-devices")
async def get_inference_devices() -> DeviceList:
    """Endpoint to get list of supported devices for inference"""
    return DeviceList(devices=inference_devices())


@device_router.get("/training-devices")
async def get_training_devices() -> DeviceList:
    """Endpoint to get list of supported devices for training"""
    return DeviceList(devices=training_devices())
