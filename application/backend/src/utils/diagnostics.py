# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing as mp
import os
import threading

logger = logging.getLogger(__name__)


def log_threads(log_level=logging.DEBUG) -> None:  # noqa: ANN001
    """Log all the threads associated with the current process and its children"""
    this_pid = os.getpid()
    children_pids = [p.pid for p in mp.active_children()]
    for pid in (this_pid, *children_pids):
        alive_threads = [thread for thread in threading.enumerate() if thread.is_alive()]
        thread_list_msg = (
            f"Alive threads for process with pid '{pid}': "
            f"{', '.join([str((thread.name, thread.ident)) for thread in alive_threads])}"
        )
        logger.log(level=log_level, msg=thread_list_msg)
