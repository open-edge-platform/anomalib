"""Deprecation utilities for anomalib.

This module provides utilities for marking functions and classes as deprecated.
The utilities help maintain backward compatibility while encouraging users to
migrate to newer APIs.

Example:
    >>> from anomalib.utils.deprecation import deprecated

    >>> # Deprecation without any information
    >>> @deprecate
    ... def old_function():
    ...     pass

    >>> # Deprecation with replacement function
    >>> @deprecate(use="new_function")
    ... def old_function():
    ...     pass

    >>> # Deprecation with version info and replacement class
    >>> @deprecate(since="2.1.0", remove="2.5.0", use="NewClass")
    ... class OldClass:
    ...     pass

    >>> # Deprecation with only removal version
    >>> @deprecate(remove="2.5.0")
    ... def another_function():
    ...     pass

    >>> # Deprecation with only deprecation version
    >>> @deprecate(since="2.1.0")
    ... def yet_another_function():
    ...     pass
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect
import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, overload

_F = TypeVar("_F", bound=Callable[..., Any])
_T = TypeVar("_T", bound=type)


@overload
def deprecate(func: _F) -> _F: ...


@overload
def deprecate(
    *,
    since: str | None = None,
    remove: str | None = None,
    use: str | None = None,
) -> Callable[[_F | _T], _F | _T]: ...


def deprecate(
    func: Callable[..., Any] | type | None = None,
    *,
    since: str | None = None,
    remove: str | None = None,
    use: str | None = None,
) -> Any:
    """Mark a function or class as deprecated.

    This decorator will cause a warning to be emitted when the function is called or class is instantiated.

    Args:
        func (Callable | type | None, optional): The function or class to be deprecated.
            If provided, the decorator is used without arguments.
            If None, the decorator is used with arguments.
            Defaults to None.
        since (str | None, optional): Version when the function/class was deprecated (e.g. "2.1.0").
            If not provided, no deprecation version will be shown in the warning message.
            Defaults to None.
        remove (str | None, optional): Version when the function/class will be removed.
            If not provided, no removal version will be shown in the warning message.
            Defaults to None.
        use (str | None, optional): Name of the replacement function/class.
            If not provided, no replacement suggestion will be shown in the warning message.
            Defaults to None.

    Returns:
        Callable: Decorated function/class that emits a deprecation warning when used.

    Example:
        >>> # Basic deprecation without any information
        >>> @deprecate
        ... def old_function():
        ...     pass

        >>> # Deprecation with version info and replacement class
        >>> @deprecate(since="2.1.0", remove="2.5.0", use="NewClass")
        ... class OldClass:
        ...     pass

        >>> # Deprecation with only removal version
        >>> @deprecate(remove="2.5.0")
        ... def another_function():
        ...     pass

        >>> # Deprecation with only deprecation version
        >>> @deprecate(since="2.1.0")
        ... def yet_another_function():
        ...     pass
    """

    def decorator(obj: Any) -> Any:  # noqa: ANN401
        if isinstance(obj, type):
            # Handle class deprecation
            original_init = inspect.getattr_static(obj, "__init__")

            @wraps(original_init)
            def new_init(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
                msg = f"{obj.__name__} is deprecated"
                if since:
                    msg += f" since v{since}"
                if remove:
                    msg += f" and will be removed in v{remove}"
                if use:
                    msg += f". Use {use} instead"
                msg += "."
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                original_init(self, *args, **kwargs)

            setattr(obj, "__init__", new_init)  # noqa: B010
            return obj

        # Handle function deprecation
        @wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            msg = f"{obj.__name__} is deprecated"
            if since:
                msg += f" since v{since}"
            if remove:
                msg += f" and will be removed in v{remove}"
            if use:
                msg += f". Use {use} instead"
            msg += "."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return obj(*args, **kwargs)

        return wrapper

    if func is None:
        # Called as @deprecate(...)
        return decorator
    # Called as @deprecate
    return decorator(func)
