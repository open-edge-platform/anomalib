# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Generator
from contextlib import contextmanager
from uuid import UUID

from loguru import logger

from core.logging.setup import global_log_config


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
