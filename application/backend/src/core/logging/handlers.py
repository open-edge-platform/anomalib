# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import inspect
import logging
import re

from loguru import logger

# Matches progress bar output (tqdm / Rich).
# Captures the step numerator and denominator so we can detect completed epochs.
# Example: "Epoch 3/199 ━━━━━━━━━━━━━━━━━━ 4/4 0:00:10 …" → groups ("4", "4")
_PROGRESS_BAR_RE = re.compile(r"[━╸╺█▏▎▍▌▋▊▉].*?(\d+)/(\d+)\s")

# Strips ANSI escape sequences (colors, bold, cursor moves) from captured output.
# Example: "\x1b[1mTrainable params\x1b[0m" → "Trainable params"
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


class InterceptHandler(logging.Handler):
    """
    This handler intercepts standard logging calls and forwards them to loguru
    while preserving the original caller information.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class LoggerStdoutWriter:
    """Wrapper for redirecting stdout/stderr to loguru."""

    @staticmethod
    def write(msg: str) -> int:
        original_length = len(msg)
        msg = _ANSI_ESCAPE_RE.sub("", msg).rstrip("\n")
        if not msg:
            return original_length
        # For progress bars, only log the final step of each epoch (N/N).
        # Intermediate redraws (1/N, 2/N, …) fire many times per second
        # and overwhelm the async log-file queue.
        match = _PROGRESS_BAR_RE.search(msg)
        if match:
            step, total = match.group(1), match.group(2)
            if step != total:
                return original_length
        logger.info(msg)
        return original_length

    @staticmethod
    def flush() -> None:
        pass
