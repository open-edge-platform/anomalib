# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import inspect
import multiprocessing as mp
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar


class ResourceUpdateFromChildProcessError(Exception):
    """Exception raised when a child process tries to update the configuration of the parent process."""

    def __init__(self):
        super().__init__(
            "Attempted to update the configuration from a child process; only the parent process can update it."
        )


P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


def parent_process_only(func: Callable) -> Callable:
    """Decorator to ensure that a method can only be called from the parent process."""
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if mp.parent_process() is not None:
                raise ResourceUpdateFromChildProcessError
            return await func(self, *args, **kwargs)

        return async_wrapper

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if mp.parent_process() is not None:
            raise ResourceUpdateFromChildProcessError
        return func(self, *args, **kwargs)

    return wrapper
