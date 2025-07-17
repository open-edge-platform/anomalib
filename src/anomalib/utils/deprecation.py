# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Deprecation utilities for anomalib.

This module provides utilities for marking functions, classes and args as deprecated.
The utilities help maintain backward compatibility while encouraging users to
migrate to newer APIs.

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

    >>> # Deprecation with reason and custom warning category
    >>> @deprecate(since="2.1.0", reason="Performance improvements available", warning_category=FutureWarning)
    ... def old_slow_function():
    ...     pass

    >>> # Deprecation of classes
    >>> @deprecate(since="2.1.0", remove="3.0.0")
    ... class MyClass:
    ...     def __init__(self):
    ...         pass

    >>> # Deprecation of specific argument(s)
    >>> @deprecate(args={"old_param": "new_param"}, since="2.1.0", remove="3.0.0")
    ... def my_function(new_param=None, old_param=None):
    ...     # Handle the mapping logic yourself
    ...     if old_param is not None and new_param is None:
    ...         new_param = old_param
    ...     # Rest of function logic
    ...     pass

    >>> @deprecate(args={"old_param": "new_param", "two_param": "two_new_param"}, since="2.1.0")
    >>> def my_args_function(one_param, two_new_param=None, new_param=None, two_param=None, old_param=None):
    ...     # Handle each mapping individually
    ...     if old_param is not None and new_param is None:
    ...         new_param = old_param
    ...     if two_param is not None and two_new_param is None:
    ...         two_new_param = two_param
    ...     # Rest of function logic
    ...     pass

    >>> @deprecate(args={"old_arg": "new_arg"}, since="2.1.0", remove="3.0.0")
    >>> class MyClass:
    ...     def __init__(self, new_arg=None, old_arg=None):
    ...         if new_arg is None and old_arg is not None:
    ...            new_arg = old_arg
    ...         pass
"""

import inspect
import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, overload

_T = TypeVar("_T")


def _build_deprecation_msg(
    name: str,
    since: str | None,
    remove: str | None,
    use: str | None,
    reason: str | None,
    prefix: str = "",
) -> str:
    """Construct a standardized deprecation warning message.

    Args:
        name: Name of the deprecated item (function, class, or argument).
        since: Version when the item was deprecated.
        remove: Version when the item will be removed.
        use: Suggested replacement name.
        reason: Additional reason for deprecation.
        prefix: Optional prefix (e.g., 'Argument ') for clarity.

    Returns:
        A complete deprecation message string.
    """
    msg = f"{prefix}{name} is deprecated"
    if since:
        msg += f" since v{since}"
    if remove:
        msg += f" and will be removed in v{remove}"
    if use:
        msg += f". Use {use} instead"
    if reason:
        msg += f". {reason}"
    return msg + "."


def _warn_deprecated_arguments(
    args_map: dict[str, str | None],
    bound_args: inspect.BoundArguments,
    since: str | None,
    remove: str | None,
    warning_category: type[Warning],
) -> None:
    """Warn about deprecated keyword arguments.

    Args:
        args_map: Mapping of deprecated argument names to new names.
        bound_args: Bound arguments of the function or method.
        since: Version when the argument was deprecated.
        remove: Version when the argument will be removed.
        warning_category: Type of warning to emit.
    """
    for old_arg, new_arg in args_map.items():
        if old_arg in bound_args.arguments:
            msg = _build_deprecation_msg(
                old_arg,
                since,
                remove,
                use=new_arg,
                reason=None,
                prefix="Argument ",
            )
            warnings.warn(msg, warning_category, stacklevel=3)


def _wrap_deprecated_init(
    cls: type,
    since: str | None,
    remove: str | None,
    use: str | None,
    reason: str | None,
    args_map: dict[str, str | None] | None,
    warning_category: type[Warning],
) -> type:
    """Wrap a class constructor to emit a deprecation warning when instantiated.

    Args:
        cls: The class being deprecated.
        since: Version when deprecation began.
        remove: Version when the class will be removed.
        use: Suggested replacement class.
        reason: Additional reason for deprecation.
        args_map: Mapping of deprecated argument names to new names.
        warning_category: The type of warning to emit.

    Returns:
        The decorated class with a warning-emitting __init__.
    """
    original_init = inspect.getattr_static(cls, "__init__")
    sig = inspect.signature(original_init)

    @wraps(original_init)
    def new_init(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        bound_args = sig.bind_partial(self, *args, **kwargs)
        bound_args.apply_defaults()

        if args_map:
            _warn_deprecated_arguments(args_map, bound_args, since, remove, warning_category)
        else:
            msg = _build_deprecation_msg(cls.__name__, since, remove, use, reason)
            warnings.warn(msg, warning_category, stacklevel=2)

        original_init(self, *args, **kwargs)

    setattr(cls, "__init__", new_init)  # noqa: B010
    return cls


def _wrap_deprecated_function(
    func: Callable,
    since: str | None,
    remove: str | None,
    use: str | None,
    reason: str | None,
    args_map: dict[str, str | None] | None,
    warning_category: type[Warning],
) -> Callable:
    """Wrap a function to emit a deprecation warning when called.

    Also handles deprecated keyword arguments, warning when old arguments are used.

    Args:
        func: The function to wrap.
        since: Version when the function was deprecated.
        remove: Version when the function will be removed.
        use: Suggested replacement function.
        reason: Additional reason for deprecation.
        args_map: Mapping of deprecated argument names to new names.
        warning_category: The type of warning to emit.

    Returns:
        The wrapped function that emits deprecation warnings when called.
    """
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        bound_args = sig.bind_partial(*args, **kwargs)
        bound_args.apply_defaults()

        # Check for deprecated arguments and warn (but don't replace)
        if args_map:
            _warn_deprecated_arguments(args_map, bound_args, since, remove, warning_category)

        # If args_map exists, this is argument deprecation, not function deprecation
        if not args_map:
            msg = _build_deprecation_msg(func.__name__, since, remove, use, reason)
            warnings.warn(msg, warning_category, stacklevel=2)

        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper


@overload
def deprecate(__obj: _T) -> _T: ...


@overload
def deprecate(
    *,
    since: str | None = None,
    remove: str | None = None,
    use: str | None = None,
    reason: str | None = None,
    args: dict[str, str | None] | None = None,
    warning_category: type[Warning] = FutureWarning,
) -> Callable[[_T], _T]: ...


def deprecate(
    __obj: Any = None,
    *,
    since: str | None = None,
    remove: str | None = None,
    use: str | None = None,
    reason: str | None = None,
    args: dict[str, str | None] | None = None,
    warning_category: type[Warning] = DeprecationWarning,
) -> Any:
    """Mark a function, class, or keyword argument as deprecated.

    This decorator will cause a warning to be emitted when the function is called,
    class is instantiated, or deprecated keyword arguments are used.
    If args are passed, no function deprecation will be shown.

    Args:
        __obj: The function or class to be deprecated.
            If provided, the decorator is used without arguments.
            If None, the decorator is used with arguments.
        since: Version when the function/class/arg was deprecated (e.g. "2.1.0").
            If not provided, no deprecation version will be shown in the warning message.
        remove: Version when the function/class/arg will be removed.
            If not provided, no removal version will be shown in the warning message.
        use: Name of the replacement function/class/arg.
            If not provided, no replacement suggestion will be shown in the warning message.
        reason: Additional reason for the deprecation.
            If not provided, no reason will be shown in the warning message.
        args: Mapping of deprecated argument names to their replacements.
            If not provided, no argument deprecation warning will be shown.
            If provided, function deprecation will not be show.
        warning_category: Type of warning to emit (default: DeprecationWarning).
            Can be DeprecationWarning, FutureWarning, PendingDeprecationWarning, etc.

    Returns:
        Decorated function/class that emits a deprecation warning when used.

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

        >>> # Deprecation with reason and custom warning category
        >>> @deprecate(since="2.1.0", reason="Performance improvements available", warning_category=FutureWarning)
        ... def old_slow_function():
        ...     pass

        >>> # Deprecation of specific argument(s)
        >>> @deprecate(args={"old_param": "new_param"}, since="2.1.0", remove="3.0.0")
        ... def my_function(new_param=None, old_param=None):
        ...     # Handle the mapping logic yourself
        ...     if old_param is not None and new_param is None:
        ...         new_param = old_param
        ...     # Rest of function logic
        ...     pass

        >>> @deprecate(args={"old_param": "new_param", "two_param": "two_new_param"}, since="2.1.0")
        >>> def my_args_function(one_param, two_new_param=None, new_param=None, two_param=None, old_param=None):
        ...     # Handle each mapping individually
        ...     if old_param is not None and new_param is None:
        ...         new_param = old_param
        ...     if two_param is not None and two_new_param is None:
        ...         two_new_param = two_param
        ...     # Rest of function logic
        ...     pass
        >>> @deprecate(args={"old_arg": "new_arg"}, since="2.1.0", remove="3.0.0")
        >>> class MyClass:
        >>>     def __init__(self, new_arg=None, old_arg=None):
        >>>         if new_arg is None and old_arg is not None:
        >>>            new_arg = old_arg
    """

    def _deprecate_impl(obj: Any) -> Any:  # noqa: ANN401
        if isinstance(obj, type):
            return _wrap_deprecated_init(obj, since, remove, use, reason, args, warning_category)
        return _wrap_deprecated_function(obj, since, remove, use, reason, args, warning_category)

    return _deprecate_impl if __obj is None else _deprecate_impl(__obj)
