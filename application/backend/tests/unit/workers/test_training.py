# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workers.training import _train_loop


class TestTrainLoop:
    """Test cases for the _train_loop async function."""

    @pytest.fixture
    def mock_stop_event(self):
        """Fixture for a mock stop event."""
        event = MagicMock()
        event.is_set.return_value = False
        return event

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
                # Set up stop event to trigger after a short time
                async def set_stop_event():
                    await asyncio.sleep(0.1)
                    mock_stop_event.is_set.return_value = True

                # Start the stop event task
                stop_task = asyncio.create_task(set_stop_event())

                # Run the train loop
                await _train_loop(mock_stop_event)

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
                # Set up stop event to trigger after a short time
                async def set_stop_event():
                    await asyncio.sleep(0.1)
                    mock_stop_event.is_set.return_value = True

                # Start the stop event task
                stop_task = asyncio.create_task(set_stop_event())

                # Run the train loop - should not raise exception
                await _train_loop(mock_stop_event)

                # Wait for stop task to complete
                await stop_task

                # Verify training service was called despite exception
                mock_training_service.train_pending_job.assert_called()

        asyncio.run(run_test())

    def test_shutdown_behavior(self, mock_stop_event, mock_training_service):
        """Test that loop responds to stop_event and exits gracefully."""

        async def run_test():
            # Set stop event immediately
            mock_stop_event.is_set.return_value = True

            with patch("workers.training.TrainingService", return_value=mock_training_service):
                # Run the train loop
                await _train_loop(mock_stop_event)

                # Verify training service was never called (loop exited immediately)
                mock_training_service.train_pending_job.assert_not_called()

        asyncio.run(run_test())
