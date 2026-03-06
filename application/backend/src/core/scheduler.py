# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import asyncio
import atexit
import multiprocessing as mp
import os
import queue
import threading  # noqa: TC003
from multiprocessing.shared_memory import SharedMemory

import psutil
from loguru import logger

from core.mjpeg_broadcaster import MJPEGBroadcaster
from services.metrics_service import SIZE
from utils.singleton import Singleton
from workers import DispatchingWorker, InferenceWorker, StreamLoader, TrainingWorker


class Scheduler(metaclass=Singleton):
    """Manages application processes and threads"""

    FRAME_QUEUE_SIZE = 5
    PREDICTION_QUEUE_SIZE = 5

    def __init__(self) -> None:
        logger.info("Initializing Scheduler...")
        # Queue for the frames acquired from the stream source and decoded
        self.frame_queue: mp.Queue = mp.Queue(maxsize=self.FRAME_QUEUE_SIZE)
        # Queue for the inference results (predictions)
        self.pred_queue: mp.Queue = mp.Queue(maxsize=self.PREDICTION_QUEUE_SIZE)
        # Queue for pushing predictions to the visualization stream (WebRTC)
        self.rtc_stream_queue: queue.Queue = queue.Queue(maxsize=1)
        # Broadcaster for MJPEG stream consumers
        self.mjpeg_broadcaster = MJPEGBroadcaster()
        # Event to sync all processes on application shutdown
        self.mp_stop_event = mp.Event()
        # Event to signal that the model has to be reloaded
        self.mp_model_reload_event = mp.Event()
        # Condition variable to notify processes about configuration updates
        self.mp_config_changed_condition = mp.Condition()

        # Shared memory for metrics collector
        self.shm_metrics: SharedMemory = SharedMemory(create=True, size=SIZE)
        self.shm_metrics_lock = mp.Lock()

        self.processes: list[mp.Process] = []
        self.threads: list[threading.Thread] = []
        logger.info("Scheduler initialized")
        # Ensure we always attempt a graceful shutdown when the main process exits
        atexit.register(self.shutdown)

    def initialize_broadcaster(self, loop: asyncio.AbstractEventLoop) -> None:
        """Initialize the MJPEG broadcaster with the event loop.

        Args:
            loop (asyncio.AbstractEventLoop): Event loop used for async broadcasts.
        """
        self.mjpeg_broadcaster.initialize(loop)

    def start_workers(self) -> None:
        """Start all worker processes and threads"""
        logger.info("Starting worker processes...")

        # Create and start processes
        training_proc = TrainingWorker(
            stop_event=self.mp_stop_event,
            logger_=logger.bind(worker=TrainingWorker.__name__),
        )
        # Training worker is not a daemon so that training script can spawn child processes
        training_proc.daemon = False

        # Inference worker consumes frames and produces predictions

        inference_proc = InferenceWorker(
            frame_queue=self.frame_queue,
            pred_queue=self.pred_queue,
            stop_event=self.mp_stop_event,
            model_reload_event=self.mp_model_reload_event,
            shm_name=self.shm_metrics.name,
            shm_lock=self.shm_metrics_lock,
            logger_=logger.bind(worker=InferenceWorker.__name__),
        )
        inference_proc.daemon = True

        # Dispatching worker consumes predictions and publishes to outputs/WebRTC
        dispatching_thread = DispatchingWorker(
            pred_queue=self.pred_queue,
            rtc_stream_queue=self.rtc_stream_queue,
            mjpeg_broadcaster=self.mjpeg_broadcaster,
            stop_event=self.mp_stop_event,
            active_config_changed_condition=self.mp_config_changed_condition,
        )

        stream_loader_proc = StreamLoader(
            frame_queue=self.frame_queue,
            stop_event=self.mp_stop_event,
            config_changed_condition=self.mp_config_changed_condition,
            logger_=logger.bind(worker=StreamLoader.__name__),
        )
        stream_loader_proc.daemon = True

        # Start all workers
        training_proc.start()
        inference_proc.start()
        stream_loader_proc.start()
        dispatching_thread.daemon = True
        dispatching_thread.start()

        # Track processes and threads
        self.processes.extend([training_proc, inference_proc, stream_loader_proc])
        self.threads.append(dispatching_thread)

        logger.info("All worker processes started successfully")

    def shutdown(self) -> None:
        """Shutdown all processes gracefully"""
        logger.info("Initiating graceful shutdown...")

        # Signal all processes to stop
        self.mp_stop_event.set()
        self.mjpeg_broadcaster.shutdown()

        # Get current process info for debugging
        pid = os.getpid()
        cur_process = psutil.Process(pid)
        alive_children = [child.pid for child in cur_process.children(recursive=True) if child.is_running()]
        logger.debug(f"Alive children of process '{pid}': {alive_children}")

        # Join threads first
        for thread in self.threads:
            if thread.is_alive():
                logger.debug(f"Joining thread: {thread.name}")
                thread.join(timeout=10)

        # Join processes in reverse order so that consumers are terminated before producers.
        for process in self.processes[::-1]:
            if process.is_alive():
                logger.debug(f"Joining process: {process.name}")
                process.join(timeout=10)
                if process.is_alive():
                    logger.warning(f"Force terminating process: {process.name}")
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        logger.error(f"Force killing process {process.name}")
                        process.kill()
                # Explicitly close the process' resources
                try:
                    process.close()
                except Exception as e:
                    logger.warning(f"Error closing process {process.name}: {e}")

        logger.info("All workers shut down gracefully")

        # Clear references
        self.processes.clear()
        self.threads.clear()

        self._cleanup_queues()
        self._cleanup_shared_memory()

    def _cleanup_queues(self) -> None:
        """Final queue cleanup"""
        for q, name in [(self.frame_queue, "frame_queue"), (self.pred_queue, "pred_queue")]:
            if q is not None:
                try:
                    # https://runebook.dev/en/articles/python/library/multiprocessing/multiprocessing.Queue.close
                    q.close()
                    q.join_thread()
                    logger.debug(f"Successfully cleaned up {name}")
                except Exception as e:
                    logger.warning(f"Error cleaning up {name}: {e}")

    def _cleanup_shared_memory(self) -> None:
        """Clean up shared memory objects"""
        if hasattr(self, "shm_metrics"):
            try:
                self.shm_metrics.close()
                self.shm_metrics.unlink()  # Remove the shared memory segment
                # Delete the attribute to make shutdown idempotent and prevent accidental reuse
                del self.shm_metrics
                logger.debug("Successfully cleaned up shared memory")
            except Exception as e:
                logger.warning(f"Error cleaning up shared memory: {e}")
