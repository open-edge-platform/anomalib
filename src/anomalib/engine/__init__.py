# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Engine module for training and evaluating anomaly detection models.

This module provides functionality for training and evaluating anomaly detection
models. The main component is the :class:`Engine` class which handles:

- Model training and validation
- Metrics computation and logging
- Checkpointing and model export
- Distributed training support

Example:
    Create and use an engine:

    >>> from anomalib.engine import Engine
    >>> engine = Engine()
    >>> engine.train()  # doctest: +SKIP
    >>> engine.test()  # doctest: +SKIP

    The engine can also be used with a custom configuration:

    >>> engine, model, datamodule = Engine.from_config(config_path="config.yaml") # doctest: +SKIP


"""

import importlib

__all__ = ["Engine", "SingleXPUStrategy", "XPUAccelerator"]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Engine": (".engine", "Engine"),
    "XPUAccelerator": (".accelerator", "XPUAccelerator"),
    "SingleXPUStrategy": (".strategy", "SingleXPUStrategy"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path, __name__)
        obj = getattr(mod, attr_name)
        globals()[name] = obj
        return obj
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
