# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter, HTTPException

from api.endpoints import API_PREFIX
from pydantic_models.devices import Camera, CameraList, DeviceList
from utils.devices import Devices

device_router = APIRouter(
    prefix=API_PREFIX + "/devices",
    tags=["Job"],
)


@device_router.get("/inference")
async def get_inference_devices() -> DeviceList:
    """Endpoint to get list of supported devices for inference"""
    return DeviceList(devices=Devices.inference_devices())


@device_router.get("/training")
async def get_training_devices() -> DeviceList:
    """Endpoint to get list of supported devices for training"""
    return DeviceList(devices=Devices.training_devices())


@device_router.get("/camera")
async def get_camera_devices() -> CameraList:
    """Endpoint to get list of available camera devices."""
    try:
        result = Devices.get_camera_devices()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    camera_list = [Camera(index=cam["index"], name=cam["name"]) for cam in result]
    return CameraList(devices=camera_list)
