# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Logging configuration and utilities for the application.

Provides centralized logging using loguru with:
- Worker-specific log files (training, inference, dispatching, stream loading)
- Job-specific temporary log files via context managers
- Automatic log rotation (10MB) and retention (10 days)
- JSON serialization and thread-safe async logging
"""

import logging
import os
import sys
from typing import TYPE_CHECKING

from core.logging.handlers import InterceptHandler
from core.logging.log_config import LogConfig

if TYPE_CHECKING:
    from loguru import Record

import pathlib

from loguru import logger

global_log_config = LogConfig()


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
    global global_log_config  # noqa: PLW0603

    if config is None:
        config = LogConfig()

    # overwrite global log_config
    global_log_config = config

    logger.remove()
    logger.add(sys.stderr, level=global_log_config.level)

    for worker_name, log_file in global_log_config.worker_log_info.items():

        def worker_log_filter(record: "Record", worker: str | None = worker_name) -> bool:
            return record["extra"].get("worker") == worker

        log_path = os.path.join(config.log_folder, log_file)

        try:
            pathlib.Path(os.path.dirname(log_path)).mkdir(exist_ok=True, parents=True)
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


def setup_uvicorn_logging() -> None:
    """Configure uvicorn logging to be handled by loguru.

    Intercepts all uvicorn log messages (from uvicorn.error, uvicorn.access, etc.)
    and redirects them to loguru for unified logging output. This ensures uvicorn
    logs follow the same format and routing as application logs.

    The function:
    1. Attaches InterceptHandler to the main uvicorn logger
    2. Sets propagate=False on uvicorn logger to prevent duplicate logs to root
    3. Clears handlers from child loggers (uvicorn.access, uvicorn.error)
    4. Enables propagation on child loggers so they forward to parent uvicorn logger

    Note: This should be called during application startup, typically before
    starting the uvicorn server.

    Example:
        >>> setup_uvicorn_logging()
        # All uvicorn logs now flow through loguru
    """
    # Setup uvicorn logs to be handled by loguru
    # Configure the main uvicorn logger with InterceptHandler
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = [InterceptHandler()]
    uvicorn_logger.setLevel(logging.INFO)
    uvicorn_logger.propagate = False  # Don't propagate to root to avoid duplicate logs
    # Clear handlers from child loggers and let them propagate to parent uvicorn logger
    for logger_name in ("uvicorn.access", "uvicorn.error"):
        child_logger = logging.getLogger(logger_name)
        child_logger.handlers.clear()
        child_logger.propagate = True  # Propagate to parent uvicorn logger
