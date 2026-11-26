# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import multiprocessing as mp
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workers.training import TrainingWorker


class TestTrainingWorker:
    """Test cases for the TrainingWorker class."""

    @pytest.fixture
    def mock_stop_event(self):
        """Fixture for a mock stop event."""
        return mp.Event()

    @pytest.fixture
    def mock_training_service(self):
        """Fixture for a mock training service."""
        service = MagicMock()
        service.train_pending_job = AsyncMock(return_value=None)
        return service

    def test_basic_operation(self, mock_stop_event, mock_training_service):
        """Test basic loop operation - runs and calls training service."""

        async def run_test():
            with patch("workers.training.TrainingService", return_value=mock_training_service):
                worker = TrainingWorker(stop_event=mock_stop_event)

                # Set up stop event to trigger after a short time
                async def set_stop_event():
                    await asyncio.sleep(0.1)
                    mock_stop_event.set()

                # Start the stop event task
                stop_task = asyncio.create_task(set_stop_event())

                # Run the train loop
                await worker.run_loop()

                # Wait for stop task to complete
                await stop_task

                # Verify training service was called
                mock_training_service.train_pending_job.assert_called()

        asyncio.run(run_test())

    def test_handles_exceptions(self, mock_stop_event, mock_training_service):
        """Test that loop handles exceptions gracefully and continues running."""

        async def run_test():
            # Mock training service to raise an exception
            mock_training_service.train_pending_job.side_effect = Exception("Training failed")

            with patch("workers.training.TrainingService", return_value=mock_training_service):
                worker = TrainingWorker(stop_event=mock_stop_event)

                # Set up stop event to trigger after a short time
                async def set_stop_event():
                    await asyncio.sleep(0.1)
                    mock_stop_event.set()

                # Start the stop event task
                stop_task = asyncio.create_task(set_stop_event())

                # Run the train loop - should not raise exception
                await worker.run_loop()

                # Wait for stop task to complete
                await stop_task

                # Verify training service was called despite exception
                mock_training_service.train_pending_job.assert_called()

        asyncio.run(run_test())

    def test_shutdown_behavior(self, mock_stop_event, mock_training_service):
        """Test that loop responds to stop_event and exits gracefully."""

        async def run_test():
            # Set stop event immediately
            mock_stop_event.set()

            with patch("workers.training.TrainingService", return_value=mock_training_service):
                worker = TrainingWorker(stop_event=mock_stop_event)

                # Run the train loop
                await worker.run_loop()

                # Verify training service was never called (loop exited immediately)
                mock_training_service.train_pending_job.assert_not_called()

        asyncio.run(run_test())
