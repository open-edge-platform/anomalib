# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Import utilities for Anomalib.

This module provides utilities for importing modules and classes from Anomalib.
"""


class OptionalImport:
    """A simple dummy class that raises ImportError when instantiated.

    This class provides a clean way to handle optional imports by creating a
    dummy class that raises informative ImportError messages when instantiated.

    Example:
        >>> # Instead of duplicating dummy class code:
        >>> if TYPE_CHECKING or module_available("some_package"):
        ...     from some_package import SomeClass
        ... else:
        ...     class SomeClass:
        ...         def __init__(self, *args, **kwargs):
        ...             raise ImportError("Some package is not installed...")

        >>> # Use OptionalImport:
        >>> SomeClass = OptionalImport("some_package")  # Auto-generates: uv pip install some_package
        >>> # Or with custom install command:
        >>> SomeClass = OptionalImport("some_package", "pip install some-package")
    """

    def __init__(
        self,
        package_name: str,
        install_command: str | None = None,
        extra_message: str = "",
    ) -> None:
        """Initialize the OptionalImport dummy class.

        Args:
            package_name: Name of the package that should be installed.
            install_command: Command to install the package. If None, defaults to
                "uv pip install <package_name>".
            extra_message: Additional message to include in the error.

        Raises:
            ImportError: Always raised immediately upon initialization.
        """
        if install_command is None:
            install_command = f"uv pip install {package_name}"

        message = f"{package_name} is not installed. Please install it using: `{install_command}`"
        if extra_message:
            message += f" {extra_message}"
        raise ImportError(message)
