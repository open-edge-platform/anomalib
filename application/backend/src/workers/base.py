# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import abc
import asyncio
import multiprocessing as mp
import os
import signal
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Event

import loguru
from loguru import logger


def log_threads(log_level="DEBUG") -> None:  # noqa: ANN001
    """Log all the alive threads associated with the current process"""
    pid = os.getpid()
    alive_threads = [thread for thread in threading.enumerate() if thread.is_alive()]
    thread_list_msg = (
        f"Alive threads for process with pid '{pid}': "
        f"{', '.join([str((thread.name, thread.ident)) for thread in alive_threads])}"
    )
    logger.log(log_level, thread_list_msg)


class StoppableMixin:
    """Mixin providing stop-aware functionality using external stop event."""

    def should_stop(self) -> bool:
        """Check if a stop has been requested."""
        if not hasattr(self, "_stop_event"):
            raise AttributeError("StoppableMixin requires a '_stop_event' to be set.")
        # Stop if parent process died
        parent_process = mp.parent_process()
        parent_died = parent_process is not None and not parent_process.is_alive()
        return self._stop_event.is_set() or parent_died  # type: ignore

    def stop_aware_sleep(self, seconds: float) -> bool:
        """
        Sleep for the specified time, but wake up immediately if stop is requested.

        Args:
            seconds: Maximum time to sleep in seconds

        Returns:
            True if woke up due to stop request, False if timeout elapsed
        """
        if not hasattr(self, "_stop_event"):
            raise AttributeError("StoppableMixin requires _stop_event to be set")
        return self._stop_event.wait(seconds)  # type: ignore


class BaseProcessWorker(mp.Process, StoppableMixin, ABC):
    """
    Reusable worker with a clean lifecycle: setup() -> run_loop() [until stop_event] -> teardown()
    Subclasses only implement what's specific to their job.
    """

    # Override in subclasses for a nicer auto-name:
    ROLE: str = "Worker"

    def __init__(
        self,
        *,
        stop_event: Event,
        queues_to_cancel: Iterable[Queue] | None = None,
        logger_: loguru.Logger | None = None,
    ) -> None:
        super().__init__()
        self._stop_event = stop_event
        self._parent_pid = os.getpid()
        self._queues_to_cancel = list(queues_to_cancel or [])

        # Platforms that use "spawn" for multiprocessing (e.g. Windows) cause logging concurrency issues.
        # Therefore, we need to copy the logger with enqueue=True in child processes.
        # https://loguru.readthedocs.io/en/stable/resources/recipes.html#compatibility-with-multiprocessing-using-enqueue-argument
        global logger  # noqa: PLW0603
        logger = logger_ or logger

    # Hooks to be implemented by subclasses

    def setup(self) -> None:
        """Allocate resources and initialize settings. Called once in the child process."""
        # Logging needs to be re-setup in child processes because settings are non-pickable.
        # Only required when multiprocessing uses "spawn" mode (Windows/MacOS)
        from core.logging import setup_logging  # noqa: PLC0415

        setup_logging()

    @abstractmethod
    async def run_loop(self) -> None:
        """
        Main loop. Return only when asked to stop or on unrecoverable error.
        self.should_stop() should be used the loop condition.
        """
        ...

    def teardown(self) -> None:
        """Release resources (optional)."""

    # Internal + final run orchestration

    @staticmethod
    def _install_signal_policy() -> None:
        """
        Ignore shutdown signals (SIGINT) in child processes.

        This function prevents child processes from handling shutdown signals directly,
        ensuring that cleanup is coordinated through the parent process via the stop_event
        mechanism.
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _auto_name(self) -> str:
        """Generate a name for the process based on its role and PIDs."""
        return "-".join([self.ROLE, str(self._parent_pid), str(os.getpid())])

    def _cancel_queue_join_threads(self) -> None:
        for q in self._queues_to_cancel:
            try:
                # https://docs.python.org/3/library/multiprocessing.html#all-start-methods
                # section: Joining processes that use queues
                # Call cancel_join_thread() to prevent the parent process from blocking
                # indefinitely when joining child processes that used this queue. This avoids potential
                # deadlocks if the queue's background thread adds more items during the flush.
                q.cancel_join_thread()
                logger.debug(f"Cancelled join thread for queue {getattr(q, 'name', q)!r}")
            except Exception as e:
                logger.warning(f"Failed cancelling queue join thread: {e}")

    def run(self) -> None:  # final orchestration; should not be overridden in subclasses
        self.setup()
        with logger.contextualize(worker=self.__class__.__name__):
            self._install_signal_policy()
            self.name = self._auto_name()
            logger.info(f"Starting {self.name}...")
            try:
                asyncio.run(self.run_loop())
            finally:
                try:
                    self.teardown()
                finally:
                    # always try to cancel any declared queues
                    self._cancel_queue_join_threads()
                    log_threads()
                    logger.info(f"Stopped {self.name}.")


class BaseThreadWorker(threading.Thread, StoppableMixin, abc.ABC):
    ROLE: str = "Worker"

    def __init__(self, *, stop_event: Event, daemon: bool = False):
        super().__init__(daemon=daemon)
        self._stop_event = stop_event
        self.name = f"{self.ROLE}-{os.getpid()}-thread"

    # hooks
    def setup(self) -> None:
        pass

    @abstractmethod
    async def run_loop(self) -> None: ...

    def teardown(self) -> None:
        pass

    # final run orchestration
    def run(self) -> None:  # final orchestrator
        with logger.contextualize(worker=self.__class__.__name__):
            logger.info(f"Starting {self.name}")
            try:
                self.setup()
                asyncio.run(self.run_loop())
            except Exception as e:
                logger.error(f"Unhandled exception in {self.name}: {str(e)}")
            finally:
                try:
                    self.teardown()
                finally:
                    logger.info(f"Stopped {self.name}")
