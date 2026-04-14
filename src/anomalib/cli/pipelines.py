# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomalib pipeline subcommands.

This module provides functionality for managing and running Anomalib pipelines through
the CLI. It includes support for benchmarking and other pipeline operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import importlib.util

if TYPE_CHECKING:
    from jsonargparse import Namespace

    from anomalib.pipelines.components.base import Pipeline

logger = logging.getLogger(__name__)

_UNINITIALIZED = object()

_PIPELINE_REGISTRY: dict[str, type[Pipeline]] | None | object = _UNINITIALIZED

_PIPELINE_DESCRIPTIONS: dict[str, str] = {
    "benchmark": "Benchmarking pipeline for evaluating anomaly detection models.",
}


def _ensure_registry() -> dict[str, type[Pipeline]] | None:
    global _PIPELINE_REGISTRY  # noqa: PLW0603
    if _PIPELINE_REGISTRY is _UNINITIALIZED:
        if importlib.util.find_spec("anomalib.pipelines") is not None:
            from anomalib.pipelines import Benchmark

            _PIPELINE_REGISTRY = {"benchmark": Benchmark}
        else:
            _PIPELINE_REGISTRY = None
    return _PIPELINE_REGISTRY  # type: ignore[return-value]


def __getattr__(name: str) -> object:
    if name == "PIPELINE_REGISTRY":
        return _ensure_registry()
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def pipeline_subcommands() -> dict[str, dict[str, str]]:
    """Get available pipeline subcommands.

    Returns:
        dict[str, dict[str, str]]: Dictionary mapping subcommand names to their descriptions.

    Example:
        Pipeline subcommands are available only if the pipelines are installed::

        >>> pipeline_subcommands()
        {
            'benchmark': {
                'description': 'Run benchmarking pipeline for model evaluation'
            }
        }
    """
    if importlib.util.find_spec("anomalib.pipelines") is None:
        return {}
    return {name: {"description": desc} for name, desc in _PIPELINE_DESCRIPTIONS.items()}


def run_pipeline(args: Namespace) -> None:
    """Run a pipeline with the provided arguments.

    Args:
        args (Namespace): Arguments for the pipeline, including the subcommand
            and configuration.

    Raises:
        ValueError: If pipelines are not available in the current installation.

    Note:
        This feature is experimental and may change or be removed in future versions.
    """
    logger.warning("This feature is experimental. It may change or be removed in the future.")
    registry = _ensure_registry()
    if registry is not None:
        subcommand = args.subcommand
        config = args[subcommand]
        registry[subcommand]().run(config)
    else:
        msg = "Pipeline is not available"
        raise ValueError(msg)
