# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from settings import get_settings
from workers import DispatchingWorker, InferenceWorker, StreamLoader, TrainingWorker

settings = get_settings()
WORKERS_FOLDER = os.path.join(settings.log_dir, "workers")
JOBS_FOLDER = os.path.join(settings.log_dir, "jobs")


@dataclass
class LogConfig:
    """Configuration for logging behavior."""

    rotation: str = "10 MB"
    retention: str = "10 days"
    level: str = "DEBUG" if settings.debug else "INFO"
    serialize: bool = True
    log_folder: Path = settings.log_dir
    # Mapping of worker classes to their dedicated log files
    # None key is used for application-level logs that don't belong to any specific worker
    worker_log_info: ClassVar[dict[str | None, str]] = {
        TrainingWorker.__name__: "training.log",
        InferenceWorker.__name__: "inference.log",
        DispatchingWorker.__name__: "dispatching.log",
        StreamLoader.__name__: "stream_loader.log",
        None: "app.log",
    }
    tensorboard_log_path: str = os.path.join(settings.log_dir, "tensorboard")
