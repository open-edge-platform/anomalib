# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing as mp
import time
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.synchronize import Lock

from utils import log_threads, suppress_child_shutdown_signals

logger = logging.getLogger(__name__)


def inference_routine(
    frame_queue: mp.Queue,  # noqa: ARG001
    pred_queue: mp.Queue,
    stop_event: EventClass,
    model_reload_event: EventClass,  # noqa: ARG001
    shm_name: str,  # noqa: ARG001
    shm_lock: Lock,  # noqa: ARG001
) -> None:
    """Load frames from the frame queue, run inference then inject the result into the predictions queue"""
    suppress_child_shutdown_signals()

    try:
        while not stop_event.is_set():
            # Get the model, reloading it if necessary
            time.sleep(10)
            print("inference_routine")
    finally:
        # https://docs.python.org/3/library/multiprocessing.html#all-start-methods
        # section: Joining processes that use queues
        # Call cancel_join_thread() to prevent the parent process from blocking
        # indefinitely when joining child processes that used this queue. This avoids potential
        # deadlocks if the queue's background thread adds more items during the flush.
        if pred_queue is not None:
            logger.debug("Cancelling the pred_queue join thread to allow inference process to exit")
            pred_queue.cancel_join_thread()

        log_threads(log_level=logging.DEBUG)
        logger.info("Stopped inference routine")
    # model.inference_adapter.await_any()
