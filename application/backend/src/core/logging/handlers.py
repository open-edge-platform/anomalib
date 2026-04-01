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

# Strips ANSI escape sequences (colors, bold, cursor hide/show, etc.) from captured output.
# Includes CSI sequences with '?' (e.g., \x1b[?25l cursor hide) emitted by tqdm/Rich.
# Example: "\x1b[1mTrainable params\x1b[0m" → "Trainable params"
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")


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
        msg = _ANSI_ESCAPE_RE.sub("", msg)
        # tqdm/Rich progress bars rely heavily on carriage returns to redraw a single line in-place.
        # When captured via stdout redirection, those \r characters leak into log files and produce
        # visually broken output, so we strip them before any further processing.
        msg = msg.replace("\r", "").rstrip("\n")
        if not msg.strip():
            return original_length
        # Collapse rich/tqdm progress bars into a compact textual summary so that logs still capture
        # coarse-grained progress (e.g., "1/4 epochs", "14% weights downloaded") without drawing the
        # full bar or leaking control characters into log files.
        match = _PROGRESS_BAR_RE.search(msg)
        if match:
            step_str, total_str = match.group(1), match.group(2)
            try:
                step = int(step_str)
                total = int(total_str)
                pct = int(step * 100 / total) if total else 0
                replacement = f" {step}/{total} ({pct}%) "
            except ValueError:
                # Fallback: keep the raw ratio if parsing fails.
                replacement = f" {step_str}/{total_str} "
            msg = _PROGRESS_BAR_RE.sub(replacement, msg)
        logger.info(msg.strip())
        return original_length

    @staticmethod
    def flush() -> None:
        pass
