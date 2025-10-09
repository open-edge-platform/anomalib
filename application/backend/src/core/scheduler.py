# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import atexit
import multiprocessing as mp
import os
import queue
import threading
from multiprocessing.shared_memory import SharedMemory

import psutil

from services.metrics_service import SIZE
from utils.singleton import Singleton
from workers import dispatching_routine, inference_routine, training_routine
from workers.stream_loading import frame_acquisition_routine

logger = logging.getLogger(__name__)


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
        # Event to sync all processes on application shutdown
        self.mp_stop_event = mp.Event()
        # Event to signal that the model has to be reloaded
        self.mp_model_reload_event = mp.Event()
        # Condition variable to notify processes about configuration updates
        self.mp_config_changed_condition = mp.Condition()

        # Shared memory for metrics collector
        self.shm_metrics = SharedMemory(create=True, size=SIZE)
        self.shm_metrics_lock = mp.Lock()

        self.processes: list[mp.Process] = []
        self.threads: list[threading.Thread] = []
        logger.info("Scheduler initialized")
        # Ensure we always attempt a graceful shutdown when the main process exits
        atexit.register(self.shutdown)

    def start_workers(self) -> None:
        """Start all worker processes and threads"""
        logger.info("Starting worker processes...")

        # Create and start processes
        training_proc = mp.Process(
            target=training_routine,
            name="Training worker",
            args=(self.mp_stop_event,),
        )
        # Training worker is not a daemon so that training script can spawn child processes
        training_proc.daemon = False

        # Inference worker consumes frames and produces predictions
        inference_proc = mp.Process(
            target=inference_routine,
            name="Inference worker",
            args=(
                self.frame_queue,
                self.pred_queue,
                self.mp_stop_event,
                self.mp_model_reload_event,
                self.shm_metrics.name,
                self.shm_metrics_lock,
            ),
        )
        inference_proc.daemon = True

        # Dispatching worker consumes predictions and publishes to outputs/WebRTC
        dispatching_thread = threading.Thread(
            target=dispatching_routine,
            name="Dispatching thread",
            args=(self.pred_queue, self.rtc_stream_queue, self.mp_stop_event, self.mp_config_changed_condition),
        )

        stream_loader_proc = mp.Process(
            target=frame_acquisition_routine,
            name="Stream loader worker",
            args=(self.frame_queue, self.mp_stop_event, self.mp_config_changed_condition),
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
                    logger.warning("Force terminating process: %s", process.name)
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        logger.error("Force killing process %s", process.name)
                        process.kill()
                # Explicitly close the process' resources
                try:
                    process.close()
                except Exception as e:
                    logger.warning("Error closing process %s: %s", process.name, e)

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
                    logger.debug("Successfully cleaned up %s", name)
                except Exception as e:
                    logger.warning("Error cleaning up %s: %s", name, e)

    def _cleanup_shared_memory(self) -> None:
        """Clean up shared memory objects"""
        if hasattr(self, "shm_metrics") and self.shm_metrics is not None:
            try:
                self.shm_metrics.close()
                self.shm_metrics.unlink()  # Remove the shared memory segment
                # Clear the Python handle to make shutdown idempotent and prevent accidental reuse
                self.shm_metrics = None
                logger.debug("Successfully cleaned up shared memory")
            except Exception as e:
                logger.warning("Error cleaning up shared memory: %s", e)
