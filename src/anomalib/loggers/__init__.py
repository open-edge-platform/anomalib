# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Logging configuration and PyTorch Lightning logger integrations.

This module provides logging utilities and integrations with various logging frameworks
for use with anomaly detection models. The main components are:

- Console logging configuration via ``configure_logger()``
- Integration with logging frameworks:
    - Comet ML via :class:`AnomalibCometLogger`
    - MLflow via :class:`AnomalibMLFlowLogger`
    - TensorBoard via :class:`AnomalibTensorBoardLogger`
    - Weights & Biases via :class:`AnomalibWandbLogger`

Example:
    Configure console logging:

    >>> from anomalib.loggers import configure_logger
    >>> configure_logger(level="INFO")

    Use a specific logger:

    >>> from anomalib.loggers import AnomalibTensorBoardLogger
    >>> logger = AnomalibTensorBoardLogger(log_dir="logs")
"""

import logging

from rich.logging import RichHandler

__all__ = ["configure_logger"]

_LOGGER_NAMES = {
    "AnomalibCometLogger": ".comet",
    "AnomalibMLFlowLogger": ".mlflow",
    "AnomalibTensorBoardLogger": ".tensorboard",
    "AnomalibWandbLogger": ".wandb",
}


def __getattr__(name: str) -> object:
    if name in _LOGGER_NAMES:
        import importlib

        module = importlib.import_module(_LOGGER_NAMES[name], __name__)
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def configure_logger(level: int | str = logging.INFO) -> None:
    """Configure console logging with consistent formatting.

    This function sets up console logging with a standardized format and rich
    tracebacks. It configures both the root logger and PyTorch Lightning logger
    to use the same formatting.

    Args:
        level (int | str): Logging level to use. Can be either a string name like
            ``"INFO"`` or an integer constant like ``logging.INFO``. Defaults to
            ``logging.INFO``.

    Example:
        >>> from anomalib.loggers import configure_logger
        >>> configure_logger(level="DEBUG")  # doctest: +SKIP
        >>> logger = logging.getLogger("my_logger")
        >>> logger.info("Test message")  # doctest: +SKIP
        2024-01-01 12:00:00 - my_logger - INFO - Test message
    """
    if isinstance(level, str):
        level = logging.getLevelName(level)

    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=format_string, level=level)
    logging.getLogger().addHandler(RichHandler(rich_tracebacks=True))

    # Set Pytorch Lightning logs to have a the consistent formatting with anomalib.
    for handler in logging.getLogger("lightning.pytorch").handlers:
        handler.setFormatter(logging.Formatter(format_string))
        handler.setLevel(level)
    logging.getLogger("lightning.pytorch").addHandler(RichHandler(rich_tracebacks=True))
