# Copyright (C) 2025 Intel Corporation
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
from pydantic_models.source import Source
from workers.stream_loading import frame_acquisition_routine


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
    config.source_type = "webcam"
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

    yield create_sample


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
        mock_config_instance.get_input_config.return_value = mock_config

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
class TestFrameAcquisition:
    """Unit tests for the frame acquisition routine"""

    @pytest.fixture(scope="session", autouse=True)
    def set_multiprocessing_start_method(self):
        # Set multiprocessing start method to 'fork' to ensure mocked objects and patches
        # from the parent process are inherited by child processes. The default 'spawn'
        # method creates isolated child processes that don't inherit mocked state.
        mp.set_start_method("fork", force=True)

    def test_queue_full(self, frame_queue, mock_stream_data, stop_event, config_changed_condition, mock_services):
        """Test that stream frames are not acquired when queue is full"""

        data1, data2 = mock_stream_data(), mock_stream_data()
        frame_queue.put(data1)
        frame_queue.put(data2)

        # Start the process
        process = mp.Process(
            target=frame_acquisition_routine, args=(frame_queue, stop_event, config_changed_condition, False)
        )
        process.start()

        # Let it run for a short time to attempt frame acquisition
        time.sleep(2)

        # Stop the process
        stop_event.set()
        process.join(timeout=1)

        queue_contents = []
        while not frame_queue.empty():
            try:
                queue_contents.append(frame_queue.get())
            except queue.Empty:
                break

        # Should still have our initial frames, proving new frames were ignored
        assert len(queue_contents) == 2
        assert all(np.array_equal(el1.frame_data, el2.frame_data) for el1, el2 in zip(queue_contents, [data1, data2]))
        assert not process.is_alive(), "Process should terminate cleanly"

    def test_queue_empty(self, frame_queue, stop_event, config_changed_condition, mock_services):
        """Test that stream frames are acquired when queue is empty"""

        # Start the process
        process = mp.Process(
            target=frame_acquisition_routine, args=(frame_queue, stop_event, config_changed_condition, False)
        )
        process.start()

        time.sleep(2)

        stop_event.set()
        process.join(timeout=1)

        assert frame_queue.qsize() == 2
        assert not process.is_alive(), "Process should terminate cleanly"

    def test_cleanup(self, frame_queue, stop_event, config_changed_condition, mock_services):
        """Test that resources has been successfully released when process finished"""

        # Start the process
        process = mp.Process(
            target=frame_acquisition_routine, args=(frame_queue, stop_event, config_changed_condition, True)
        )
        process.start()

        time.sleep(2)

        stop_event.set()
        process.join(timeout=1)

        assert not process.is_alive(), "Process should terminate cleanly"
