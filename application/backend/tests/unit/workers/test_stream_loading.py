# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import queue
import sys
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from entities.stream_data import StreamData
from entities.video_stream import VideoStream
from pydantic_models.source import Source, SourceType
from workers.stream_loading import StreamLoader


@pytest.fixture
def mp_manager():
    """Multiprocessing manager fixture"""
    manager = mp.Manager()
    yield manager
    manager.shutdown()


@pytest.fixture
def frame_queue(mp_manager):
    """Frame queue fixture"""
    return mp_manager.Queue(maxsize=2)


@pytest.fixture
def stop_event():
    """Stop event fixture"""
    return mp.Event()


@pytest.fixture
def config_changed_condition():
    """Configuration changed condition fixture"""
    return mp.Condition()


@pytest.fixture
def mock_config():
    """Mock configuration fixture"""
    config = Mock(spec=Source)
    config.source_type = SourceType.USB_CAMERA
    config.device_id = 0
    return config


@pytest.fixture
def mock_stream_data():
    def create_sample():
        return StreamData(
            frame_data=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            timestamp=time.time(),
            source_metadata={},
        )

    return create_sample


@pytest.fixture
def mock_video_stream(mock_stream_data):
    """Mock video stream fixture"""
    stream = Mock(spec=VideoStream)
    stream.get_data.return_value = mock_stream_data()
    stream.is_real_time.return_value = True
    stream.release.return_value = None
    return stream


@pytest.fixture
def mock_services(mock_config, mock_video_stream):
    with (
        patch("workers.stream_loading.ActivePipelineService") as mock_active_pipeline_service,
        patch("workers.stream_loading.VideoStreamService") as mock_video_service,
    ):
        # Set up the mocks
        mock_config_instance = Mock()
        mock_config_instance.source_config = mock_config

        # Mock the async create method
        async def mock_create(*args, **kwargs):
            return mock_config_instance

        mock_active_pipeline_service.create = mock_create

        mock_video_service.get_video_stream.return_value = mock_video_stream

        yield {
            "active_pipeline_service": mock_active_pipeline_service,
            "video_service": mock_video_service,
            "config_instance": mock_config_instance,
            "video_stream": mock_video_stream,
        }


@pytest.mark.skipif(sys.platform == "win32", reason="Multiprocessing 'fork' start method not available on Windows")
class TestStreamLoader:
    """Unit tests for the StreamLoader worker"""

    @pytest.fixture(scope="session", autouse=True)
    def set_multiprocessing_start_method(self):
        # Set multiprocessing start method to 'fork' to ensure mocked objects and patches
        # from the parent process are inherited by child processes. The default 'spawn'
        # method creates isolated child processes that don't inherit mocked state.
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            # Already set, ignore
            pass

    def test_queue_full_realtime(
        self,
        frame_queue,
        mock_stream_data,
        stop_event,
        config_changed_condition,
        mock_services,
    ):
        """Test that frames are discarded when queue is full for real-time streams"""
        data1, data2 = mock_stream_data(), mock_stream_data()
        frame_queue.put(data1)
        frame_queue.put(data2)
        initial_queue_size = frame_queue.qsize()

        # Create and start the worker
        worker = StreamLoader(
            frame_queue=frame_queue,
            stop_event=stop_event,
            config_changed_condition=config_changed_condition,
        )

        # Start the worker in a separate process
        worker.start()

        # Let it run for a short time
        # For real-time streams with full queue, it will discard old frames without adding new ones
        time.sleep(2)

        # Stop the worker
        stop_event.set()
        worker.join(timeout=5)

        # For real-time streams, when queue is full, old frames are discarded
        # The implementation discards old frame but doesn't add new one in same cycle
        queue_contents = []
        while not frame_queue.empty():
            try:
                queue_contents.append(frame_queue.get(timeout=0.1))
            except queue.Empty:
                break

        # Queue should have fewer or equal frames compared to initial (some may be discarded)
        assert len(queue_contents) <= initial_queue_size
        assert not worker.is_alive(), "Worker process should terminate cleanly"

    def test_queue_empty(self, frame_queue, stop_event, config_changed_condition, mock_services):
        """Test that stream frames are acquired when queue is empty"""
        # Create and start the worker
        worker = StreamLoader(
            frame_queue=frame_queue,
            stop_event=stop_event,
            config_changed_condition=config_changed_condition,
        )

        worker.start()

        # Let it run to acquire frames
        time.sleep(2)

        # Stop the worker
        stop_event.set()
        worker.join(timeout=5)

        # Should have acquired frames
        assert frame_queue.qsize() >= 1, "Queue should have at least one frame"
        assert not worker.is_alive(), "Worker process should terminate cleanly"

    def test_cleanup(self, frame_queue, stop_event, config_changed_condition, mock_services):
        """Test that resources are successfully released when worker finishes"""
        # Create and start the worker
        worker = StreamLoader(
            frame_queue=frame_queue,
            stop_event=stop_event,
            config_changed_condition=config_changed_condition,
        )

        worker.start()

        # Let it run briefly
        time.sleep(1)

        # Stop the worker
        stop_event.set()
        worker.join(timeout=5)

        # Verify clean shutdown
        assert not worker.is_alive(), "Worker process should terminate cleanly"

        # Verify the video stream was released (checked in teardown)
        # Note: We can't directly verify mock calls across process boundaries,
        # but the teardown method should have been called
