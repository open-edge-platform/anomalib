# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
from multiprocessing.synchronize import Event as EventClass

from services.training_service import TrainingService
from workers.base import BaseProcessWorker

logger = logging.getLogger(__name__)

MAX_CONCURRENT_TRAINING = 1
SCHEDULE_INTERVAL_SEC = 5


class TrainingWorker(BaseProcessWorker):
    def __init__(self, stop_event: EventClass):
        super().__init__(stop_event=stop_event)

    async def run_loop(self) -> None:
        """Main training loop that polls for jobs and manages concurrent training tasks."""
        training_service = TrainingService()
        running_tasks: set[asyncio.Task] = set()

        while not self.should_stop():
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
                if self.should_stop():
                    break
                await asyncio.sleep(0.5)

        # Cancel any remaining tasks on shutdown
        for task in running_tasks:
            task.cancel()
        if running_tasks:
            try:
                await asyncio.gather(*running_tasks, return_exceptions=True)
            except Exception as e:
                # Log exceptions during cancellation to ensure clean shutdown and aid debugging
                logger.error(f"Exception during task cancellation: {e}", exc_info=True)
