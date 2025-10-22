# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass

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
