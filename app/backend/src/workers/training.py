# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
from multiprocessing.synchronize import Event as EventClass
import time

from services.training_service import TrainingService
from utils import suppress_child_shutdown_signals

logger = logging.getLogger(__name__)


async def _train_loop(stop_event: EventClass) -> None:
    training_service = TrainingService()
    while not stop_event.is_set():
        try:
            await training_service.train_pending_job()
        except Exception as e:
            logger.error(f"Error occurred during training: {e}", exc_info=True)
        print("tick")
        # React quickly to shutdown: sleep in short intervals
        for _ in range(10):
            if stop_event.is_set():
                break
            await asyncio.sleep(0.5)
        print("beep")


def training_routine(
    stop_event: EventClass, cleanup: bool = True
) -> None:
    suppress_child_shutdown_signals()
    try:
        # Run one event loop for the worker process
        asyncio.run(_train_loop(stop_event))
    finally:
        if cleanup:
            _cleanup_resources()
        logger.info("Stopped training worker")


def _cleanup_resources() -> None:
    pass
