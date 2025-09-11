# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
from multiprocessing.synchronize import Event as EventClass

from services.training_service import TrainingService
from utils import suppress_child_shutdown_signals

logger = logging.getLogger(__name__)

MAX_CONCURRENT_TRAINING = 1
SCHEDULE_INTERVAL_SEC = 5


async def _train_loop(stop_event: EventClass) -> None:
    """Main training loop that polls for jobs and manages concurrent training tasks."""
    training_service = TrainingService()
    running_tasks: set[asyncio.Task] = set()

    while not stop_event.is_set():
        try:
            # Clean up completed tasks
            running_tasks = {task for task in running_tasks if not task.done()}

            # Start new training if under capacity limit
            # Using async tasks allows:
            # - Multiple training jobs to run concurrently
            # - Event loop to remain responsive for shutdown signals
            if len(running_tasks) < MAX_CONCURRENT_TRAINING:
                running_tasks.add(asyncio.create_task(training_service.train_pending_job()))
        except Exception as e:
            logger.error(f"Error occurred in training loop: {e}", exc_info=True)

        # Check for shutdown signals frequently
        for _ in range(SCHEDULE_INTERVAL_SEC * 2):
            if stop_event.is_set():
                break
            await asyncio.sleep(0.5)

    # Cancel any remaining tasks on shutdown
    for task in running_tasks:
        task.cancel()


def training_routine(stop_event: EventClass, cleanup: bool = True) -> None:
    """Entry point for the training worker process."""
    suppress_child_shutdown_signals()
    try:
        asyncio.run(_train_loop(stop_event))
    finally:
        if cleanup:
            _cleanup_resources()
        logger.info("Stopped training worker")


def _cleanup_resources() -> None:
    """Clean up resources when the worker shuts down."""
