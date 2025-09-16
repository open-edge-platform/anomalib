# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import signal


def suppress_child_shutdown_signals() -> None:
    """
    Ignore shutdown signals (SIGINT) in child processes.

    This function prevents child processes from handling shutdown signals directly,
    ensuring that cleanup is coordinated through the parent process via the stop_event
    mechanism. This avoids race conditions and ensures consistent cleanup behavior.

    Notes:
        - Only applies when using 'spawn' start method (macOS, Windows)
        - Child processes should perform resource cleanup via stop_event notification
        - Parent process remains responsible for signal handling and process termination

    Implementation:
        Child processes should call this function at startup to ignore signals
        and rely on stop_event.is_set() for graceful shutdown coordination.

    Returns:
        None

    Example:
        >>> def worker_process(stop_event, resource):
        ...     suppress_child_shutdown_signals()
        ...     try:
        ...         while not stop_event.is_set():
        ...             # perform work
        ...             pass
        ...     finally:
        ...         # Cleanup resources when stop_event is set
        ...         resource.cleanup()
    """
    if mp.get_start_method(allow_none=True) == "spawn" and mp.parent_process() is not None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
