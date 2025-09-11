# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from functools import lru_cache
from sys import platform

from utils.diagnostics import log_threads
from utils.queue import flush_queue
from utils.signal import suppress_child_shutdown_signals
from utils.singleton import Singleton
from utils.visualization import Visualizer

__all__ = ["Singleton", "Visualizer", "flush_queue", "log_threads", "suppress_child_shutdown_signals"]


@lru_cache
def is_platform_darwin() -> bool:
    return platform == "darwin"
