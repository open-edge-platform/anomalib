# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections.abc import Generator
from contextlib import ContextDecorator, contextmanager, redirect_stderr, redirect_stdout
from types import TracebackType
from typing import Self
from uuid import UUID

from loguru import logger

from core.logging.handlers import InterceptHandler, LoggerStdoutWriter
from core.logging.setup import global_log_config

# Also logs the weights if they are being downloaded.
_ML_LOGGER_NAMES = (
    "timm",
    "huggingface_hub",
)


class capture_output(ContextDecorator):
    """Redirect stdout, stderr, and ML library logging into loguru.

    Usable as a decorator or context manager. While active, all print()
    output, tqdm progress bars, and standard-logging calls from common ML
    libraries are forwarded to loguru so they appear in per-job log files.

    Example (decorator)::

        @capture_output()
        def train(model): ...

    Example (context manager)::

        with capture_output():
            engine.fit(model, datamodule)
    """

    def __enter__(self) -> Self:
        self._log_writer = LoggerStdoutWriter()
        self._stdout_ctx = redirect_stdout(self._log_writer)  # type: ignore[type-var]
        self._stderr_ctx = redirect_stderr(self._log_writer)  # type: ignore[type-var]
        self._stdout_ctx.__enter__()
        self._stderr_ctx.__enter__()

        # Intercept standard logging from ML libraries.
        self._original_handlers: dict[str, list[logging.Handler]] = {}
        self._original_levels: dict[str, int] = {}
        intercept = InterceptHandler()
        for name in _ML_LOGGER_NAMES:
            lib_logger = logging.getLogger(name)
            self._original_handlers[name] = lib_logger.handlers[:]
            self._original_levels[name] = lib_logger.level
            lib_logger.handlers = [intercept]
            lib_logger.setLevel(logging.DEBUG)
            lib_logger.propagate = False

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Restore ML library loggers.
        for name in _ML_LOGGER_NAMES:
            lib_logger = logging.getLogger(name)
            lib_logger.handlers = self._original_handlers[name]
            lib_logger.level = self._original_levels[name]
            lib_logger.propagate = True

        self._stderr_ctx.__exit__(exc_type, exc_val, exc_tb)
        self._stdout_ctx.__exit__(exc_type, exc_val, exc_tb)


def _validate_job_id(job_id: str | UUID) -> str | UUID:
    """Validate job_id to prevent path traversal attacks.

    Args:
        job_id: The job identifier to validate

    Returns:
        Validated job_id

    Raises:
        ValueError: If job_id is not a valid UUID
    """
    # Only allow alphanumeric, hyphens, underscores
    try:
        UUID(str(job_id))
    except ValueError as e:
        raise ValueError(
            f"Invalid job_id '{job_id}'. Only alphanumeric characters, hyphens, and underscores are allowed.",
        ) from e
    return job_id


def get_job_logs_path(job_id: str | UUID) -> str:
    """Get the path to the logs folder for a specific job.

    Args:
        job_id: Unique identifier for the job

    Returns:
        str: Path to the job's logs folder (logs/jobs/{job_id})

    Raises:
        ValueError: If job_id contains invalid characters

    Example:
        >>> get_job_logs_path(job_id="foo-123")
        'logs/jobs/foo-123'
    """
    job_id = _validate_job_id(job_id)
    jobs_folder = os.path.join(global_log_config.log_folder, "jobs")
    try:
        os.makedirs(jobs_folder, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Failed to create jobs log directory: {e}") from e
    return os.path.join(jobs_folder, f"{job_id}.log")


@contextmanager
def job_logging_ctx(job_id: str | UUID) -> Generator[str]:
    """Add a temporary log sink for a specific job.

    Captures all logs emitted during the context to logs/jobs/{job_id}.log.
    The sink is automatically removed on exit, but the log file persists.
    Logs also continue to go to other configured sinks.

    Args:
        job_id: Unique identifier for the job, used as the log filename

    Yields:
        str: Path to the created log file (logs/jobs/{job_id}.log)

    Raises:
        ValueError: If job_id contains invalid characters
        RuntimeError: If log directory creation or sink addition fails

    Example:
        >>> with job_logging_ctx(job_id="foo-123"):
        ...     logger.info("bar")  # All logs saved to logs/jobs/train-123.log
    """
    job_id = _validate_job_id(job_id)

    log_file = get_job_logs_path(job_id)

    try:
        sink_id = logger.add(
            log_file,
            rotation=global_log_config.rotation,
            retention=global_log_config.retention,
            level=global_log_config.level,
            serialize=global_log_config.serialize,
            enqueue=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to add log sink for job {job_id}: {e}") from e

    try:
        logger.info(f"Started logging to {log_file}")
        yield log_file
    finally:
        logger.info(f"Stopped logging to {log_file}")
        logger.remove(sink_id)
