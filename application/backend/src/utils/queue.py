# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import queue

from loguru import logger


def flush_queue(queue_obj: mp.Queue) -> None:
    """Safely flush all items from a multiprocessing queue"""
    flushed_count = 0

    while True:
        try:
            queue_obj.get_nowait()
            flushed_count += 1
        except queue.Empty:
            # Queue is empty, we're done
            break
        except (OSError, ValueError, EOFError, BrokenPipeError) as e:
            # Queue is closed/invalid or connection broken
            logger.debug(f"Queue flush stopped due to: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error during queue flush: {e}")
            break

    if flushed_count > 0:
        logger.info(f"Flushed {flushed_count} items from queue")

    # https://docs.python.org/3/library/multiprocessing.html#all-start-methods
    # section: Joining processes that use queues
    # After flushing, call cancel_join_thread() to prevent the parent process from blocking
    # indefinitely when joining child processes that used this queue. This avoids potential
    # deadlocks if the queue's background thread adds more items during the flush.
    queue_obj.cancel_join_thread()
