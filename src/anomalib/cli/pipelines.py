# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomalib pipeline subcommands.

This module provides functionality for managing and running Anomalib pipelines through
the CLI. It includes support for benchmarking and other pipeline operations.
"""

import importlib.util
import logging
from typing import Any, ClassVar

from jsonargparse import Namespace

from anomalib.cli.utils.help_formatter import get_short_docstring

logger = logging.getLogger(__name__)


class _PipelineRegistry(dict):
    """Lazy dictionary for pipeline classes to defer importing heavy dependencies."""

    _MAPPING: ClassVar[dict[str, tuple[str, str]]] = {
        "benchmark": ("anomalib.pipelines.benchmark", "Benchmark"),
    }

    @classmethod
    def registered_subcommands(cls) -> list[str]:
        """Return list of registered pipeline subcommands."""
        return list(cls._MAPPING.keys())

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        if dict.__contains__(self, key):
            return super().__getitem__(key)
        if key in self._MAPPING:
            import importlib

            module_name, class_name = self._MAPPING[key]
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)
            self[key] = cls
            return cls
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in self._MAPPING

    def get(self, key: str, default: Any = None) -> Any:  # noqa: ANN401
        try:
            return self[key]
        except KeyError:
            return default


if importlib.util.find_spec("anomalib.pipelines") is not None:
    PIPELINE_REGISTRY: dict[str, Any] | None = _PipelineRegistry()
else:
    PIPELINE_REGISTRY = None


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
    if PIPELINE_REGISTRY is not None:
        descriptions = {
            "benchmark": "Benchmarking pipeline for evaluating anomaly detection models.",
        }
        result = {}
        for name in _PipelineRegistry.registered_subcommands():
            desc = descriptions[name] if name in descriptions else get_short_docstring(PIPELINE_REGISTRY[name])
            result[name] = {"description": desc}
        return result
    return {}


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
    if PIPELINE_REGISTRY is not None:
        subcommand = args.subcommand
        config = args[subcommand]
        PIPELINE_REGISTRY[subcommand]().run(config)
    else:
        msg = "Pipeline is not available"
        raise ValueError(msg)
