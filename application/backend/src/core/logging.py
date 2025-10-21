# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Logging configuration and utilities for the application.

Provides centralized logging using loguru with:
- Worker-specific log files (training, inference, dispatching, stream loading)
- Job-specific temporary log files via context managers
- Automatic log rotation (10MB) and retention (10 days)
- JSON serialization and thread-safe async logging
"""

import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from loguru import Record

from loguru import logger

from workers import DispatchingWorker, InferenceWorker, StreamLoader, TrainingWorker

LOG_FOLDER = "logs"
WORKERS_FOLDER = os.path.join(LOG_FOLDER, "workers")
JOBS_FOLDER = os.path.join(LOG_FOLDER, "jobs")


@dataclass
class LogConfig:
    """Configuration for logging behavior."""

    rotation: str = "10 MB"
    retention: str = "10 days"
    level: str = "DEBUG"
    serialize: bool = True
    log_folder: str = LOG_FOLDER
    # Mapping of worker classes to their dedicated log files
    # None key is used for application-level logs that don't belong to any specific worker
    worker_log_info = {
        TrainingWorker.__name__: "training.log",
        InferenceWorker.__name__: "inference.log",
        DispatchingWorker.__name__: "dispatching.log",
        StreamLoader.__name__: "stream_loader.log",
        None: "app.log",
    }
    tensorboard_log_path: str = os.path.join(LOG_FOLDER, "tensorboard")


log_config = LogConfig()


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
            f"Invalid job_id '{job_id}'. Only alphanumeric characters, hyphens, and underscores are allowed."
        ) from e
    return job_id


def setup_logging(config: LogConfig | None = None) -> None:
    """Configure application-wide logging with worker-specific log files.

    Creates separate log files for each worker type (training, inference, etc.) with
    configurable rotation and retention. Logs are filtered by the 'worker' field in
    record extras and serialized as JSON.

    Args:
        config: Optional LogConfig instance. If None, uses default configuration.

    Important: Must be called in each child process (BaseProcessWorker does this
    automatically) and once at main process startup. Loguru sinks don't transfer
    to child processes.

    Example:
        >>> setup_logging()
        # Creates: logs/training.log, logs/inference.log, logs/app.log, etc.
        >>> custom_config = LogConfig(rotation="50 MB", level="INFO")
        >>> setup_logging(custom_config)
    """
    if config is None:
        config = LogConfig()

    # overwrite global log_config
    log_config = config

    for worker_name, log_file in log_config.worker_log_info.items():

        def worker_log_filter(record: "Record", worker: str | None = worker_name) -> bool:
            return record["extra"].get("worker") == worker

        log_path = os.path.join(config.log_folder, log_file)

        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        except OSError as e:
            logger.warning(f"Failed to create log directory {log_path}: {e}")
            continue

        try:
            logger.add(
                log_path,
                rotation=config.rotation,
                retention=config.retention,
                level=config.level,
                filter=worker_log_filter,
                serialize=config.serialize,
                enqueue=True,
            )
        except Exception as e:
            logger.error(f"Failed to add log sink for {worker_name}: {e}")


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
    jobs_folder = os.path.join(log_config.log_folder, "jobs")
    try:
        os.makedirs(jobs_folder, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Failed to create jobs log directory: {e}") from e
    return os.path.join(jobs_folder, f"{job_id}.log")


@contextmanager
def job_logging_ctx(job_id: str | UUID) -> Generator[str, None, None]:
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
            rotation=log_config.rotation,
            retention=log_config.retention,
            level=log_config.level,
            serialize=log_config.serialize,
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


class LoggerStdoutWriter:
    @staticmethod
    def write(msg: str) -> None:
        msg = msg.rstrip("\n")
        if msg:
            logger.info(msg)

    @staticmethod
    def flush() -> None:
        pass
