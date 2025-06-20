# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Path utilities for anomaly detection.

This module provides utilities for managing paths and directories in anomaly
detection projects. The key components include:

    - Version directory creation and management
    - Symbolic link handling
    - Path resolution and validation
    - Output filename generation

Examples:
    Test create_versioned_dir:

    >>> from anomalib.utils.path import create_versioned_dir
    >>> from pathlib import Path
    >>> # Create versioned directory
    >>> version_dir = create_versioned_dir(Path("experiments"))
    >>> version_dir.name
    'v1'

The module ensures consistent path handling by:
    - Creating incrementing version directories (v1, v2, etc.)
    - Maintaining a ``latest`` symbolic link
    - Handling both string and ``Path`` inputs
    - Providing cross-platform compatibility

Note:
    All paths are resolved to absolute paths to ensure consistent behavior
    across different working directories.
"""

import re
from pathlib import Path


def create_versioned_dir(root_dir: str | Path) -> Path:
    """Create a new version directory and update the ``latest`` symbolic link.

    This function creates a new versioned directory (e.g. ``v1``, ``v2``, etc.) inside the
    specified root directory and updates a ``latest`` symbolic link to point to it.
    The version numbers increment automatically based on existing directories.

    Args:
        root_dir (Union[str, Path]): Root directory path where version directories will be
            created. Can be provided as a string or ``Path`` object. Directory will be
            created if it doesn't exist.

    Returns:
        Path: Path to the ``latest`` symbolic link that points to the newly created
            version directory.

    Examples:
        Create first version directory:

        >>> from pathlib import Path
        >>> version_dir = create_versioned_dir(Path("experiments"))
        >>> version_dir
        PosixPath('experiments/latest')
        >>> version_dir.resolve().name  # Points to v1
        'v1'

        Create second version directory:

        >>> version_dir = create_versioned_dir("experiments")
        >>> version_dir.resolve().name  # Now points to v2
        'v2'

    Note:
        - The function resolves all paths to absolute paths
        - Creates parent directories if they don't exist
        - Handles existing symbolic links by removing and recreating them
        - Version directories follow the pattern ``v1``, ``v2``, etc.
        - The ``latest`` link always points to the most recently created version
    """
    # Compile a regular expression to match version directories
    version_pattern = re.compile(r"^v(\d+)$")

    # Resolve the path
    root_dir = Path(root_dir).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    # Find the highest existing version number
    highest_version = -1
    for version_dir in root_dir.iterdir():
        if version_dir.is_dir():
            match = version_pattern.match(version_dir.name)
            if match:
                version_number = int(match.group(1))
                highest_version = max(highest_version, version_number)

    # The new directory will have the next highest version number
    new_version_number = highest_version + 1
    new_version_dir = root_dir / f"v{new_version_number}"

    # Create the new version directory
    new_version_dir.mkdir()

    # Update the 'latest' symbolic link to point to the new version directory
    latest_link_path = root_dir / "latest"
    if latest_link_path.is_symlink() or latest_link_path.exists():
        latest_link_path.unlink()
    latest_link_path.symlink_to(new_version_dir, target_is_directory=True)

    return latest_link_path


def convert_to_snake_case(s: str) -> str:
    """Convert a string to snake case format.

    This function converts various string formats (space-separated, camelCase,
    PascalCase, etc.) to snake_case by:

    - Converting spaces and punctuation to underscores
    - Inserting underscores before capital letters
    - Converting to lowercase
    - Removing redundant underscores

    Args:
        s (str): Input string to convert to snake case.

    Returns:
        str: The input string converted to snake case format.

    Examples:
        Convert space-separated string:

        >>> convert_to_snake_case("Snake Case")
        'snake_case'

        Convert camelCase:

        >>> convert_to_snake_case("snakeCase")
        'snake_case'

        Convert PascalCase:

        >>> convert_to_snake_case("SnakeCase")
        'snake_case'

        Handle existing snake_case:

        >>> convert_to_snake_case("snake_case")
        'snake_case'

        Handle punctuation:

        >>> convert_to_snake_case("snake.case")
        'snake_case'

        >>> convert_to_snake_case("snake-case")
        'snake_case'

    Note:
        - Leading/trailing underscores are removed
        - Multiple consecutive underscores are collapsed to a single underscore
        - Punctuation marks (``.``, ``-``, ``'``) are converted to underscores
    """
    # Replace whitespace, hyphens, periods, and apostrophes with underscores
    s = re.sub(r"\s+|[-.\']", "_", s)

    # Insert underscores before capital letters (except at the beginning of the string)
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()

    # Remove leading and trailing underscores
    s = re.sub(r"^_+|_+$", "", s)

    # Replace multiple consecutive underscores with a single underscore
    return re.sub(r"__+", "_", s)


def convert_snake_to_pascal_case(snake_case: str) -> str:
    """Convert snake_case string to PascalCase.

    This function takes a string in snake_case format (words separated by underscores)
    and converts it to PascalCase format (each word capitalized and concatenated).

    Args:
        snake_case (str): Input string in snake_case format (e.g. ``"efficient_ad"``)

    Returns:
        str: Output string in PascalCase format (e.g. ``"EfficientAd"``)

    Examples:
        >>> convert_snake_to_pascal_case("efficient_ad")
        'EfficientAd'
        >>> convert_snake_to_pascal_case("patchcore")
        'Patchcore'
        >>> convert_snake_to_pascal_case("reverse_distillation")
        'ReverseDistillation'
    """
    return "".join(word.capitalize() for word in snake_case.split("_"))


def convert_to_title_case(text: str) -> str:
    """Convert text to title case, handling various text formats.

    This function converts text from various formats (regular text, snake_case, camelCase,
    PascalCase) to title case format. It preserves punctuation and handles contractions
    appropriately.

    Args:
        text (str): Input text to convert to title case. Can be in any text format like
            snake_case, camelCase, PascalCase or regular text.

    Returns:
        str: The input text converted to title case format.

    Raises:
        TypeError: If the input ``text`` is not a string.

    Examples:
        Regular text:

        >>> convert_to_title_case("the quick brown fox")
        'The Quick Brown Fox'

        Snake case:

        >>> convert_to_title_case("convert_snake_case_to_title_case")
        'Convert Snake Case To Title Case'

        Camel case:

        >>> convert_to_title_case("convertCamelCaseToTitleCase")
        'Convert Camel Case To Title Case'

        Pascal case:

        >>> convert_to_title_case("ConvertPascalCaseToTitleCase")
        'Convert Pascal Case To Title Case'

        Mixed cases:

        >>> convert_to_title_case("mixed_snake_camelCase and PascalCase")
        'Mixed Snake Camel Case And Pascal Case'

        Handling punctuation and contractions:

        >>> convert_to_title_case("what's the_weather_like? it'sSunnyToday.")
        "What's The Weather Like? It's Sunny Today."

        With numbers and special characters:

        >>> convert_to_title_case("python3.9_features and camelCaseNames")
        'Python 3.9 Features And Camel Case Names'

    Note:
        - Preserves contractions (e.g., "what's" -> "What's")
        - Handles mixed case formats in the same string
        - Maintains punctuation and spacing
        - Properly capitalizes words after numbers and special characters
    """
    if not isinstance(text, str):
        msg = "Input must be a string"
        raise TypeError(msg)

    # Handle snake_case
    text = text.replace("_", " ")

    # Handle camelCase and PascalCase
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", text)

    # Split the text into words, preserving punctuation
    words = re.findall(r"[\w']+|[.,!?;]", text)

    # Capitalize each word
    result = [word.capitalize() if word.isalpha() or "'" in word else word for word in words]

    # Join the words back together
    return " ".join(result)


def generate_output_filename(
    input_path: str | Path,
    output_path: str | Path,
    dataset_name: str | None = None,
    category: str | None = None,
    mkdir: bool = True,
) -> Path:
    """Generate an output filename based on the input path.

    This function generates an output path that preserves the directory structure after the
    dataset root, using an improved algorithm that works with any folder structure.

    Args:
        input_path (str | Path): Path to the input file.
        output_path (str | Path): Base output directory path.
        dataset_name (str | None, optional): Name of the dataset to find in the input path.
            If provided, the path structure after this dataset directory is preserved.
            If not provided or not found, uses intelligent heuristics.
            Defaults to ``None``.
        category (str | None, optional): Category name to find in the input path after
            dataset name. If provided, preserves structure after this category.
            Defaults to ``None``.
        mkdir (bool, optional): Whether to create the output directory structure.
            Defaults to ``True``.

    Returns:
        Path: Generated output file path preserving relevant directory structure.

    Examples:
        Basic usage with MVTec-style dataset:

        >>> input_path = "/data/MVTecAD/bottle/test/broken_large/000.png"
        >>> generate_output_filename(input_path, "./results", "MVTecAD", "bottle")
        PosixPath('results/test/broken_large/000.png')

        Without category preserves more structure:

        >>> generate_output_filename(input_path, "./results", "MVTecAD")
        PosixPath('results/bottle/test/broken_large/000.png')

        Works with folder datasets:

        >>> path = "/datasets/MyDataset/normal/image001.png"
        >>> generate_output_filename(path, "./output", "MyDataset")
        PosixPath('output/normal/image001.png')

        Handles custom structures gracefully:

        >>> path = "/custom/path/MyData/category/split/image.png"
        >>> generate_output_filename(path, "./out", "MyData")
        PosixPath('out/category/split/image.png')

        Auto-detection when dataset_name not provided:

        >>> path = "/any/folder/structure/normal/image.png"
        >>> generate_output_filename(path, "./out")
        PosixPath('out/normal/image.png')

        Case-insensitive matching:

        >>> path = "/data/mvtecad/BOTTLE/test/000.png"
        >>> generate_output_filename(path, "./results", "MVTecAD", "bottle")
        PosixPath('results/test/000.png')

        Relative paths:

        >>> path = "data/MVTecAD/bottle/test/000.png"
        >>> generate_output_filename(path, "./results", "MVTecAD")
        PosixPath('results/bottle/test/000.png')

    Note:
        - Uses intelligent path analysis to work with any folder structure
        - Preserves directory structure after the dataset root
        - If ``mkdir=True``, creates output directory structure if it doesn't exist
        - Original filename is always preserved in output path
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Find the base path to exclude
    exclude_base = None

    # Try to find dataset_name in path
    if dataset_name:
        for i, part in enumerate(input_path.parts):
            if part.lower() == dataset_name.lower():
                exclude_base = Path(*input_path.parts[: i + 1])
                break

    # Try to find category after dataset
    if exclude_base and category:
        try:
            remaining = input_path.relative_to(exclude_base)
            for j, part in enumerate(remaining.parts):
                if part.lower() == category.lower():
                    exclude_base = exclude_base / Path(*remaining.parts[: j + 1])
                    break
        except ValueError:
            pass  # relative_to failed, keep original exclude_base

    # Use relative_to to get the remaining path structure
    if exclude_base:
        try:
            relative_path = input_path.relative_to(exclude_base)
            preserved_dirs = relative_path.parts[:-1]  # All dirs except filename
        except ValueError:
            # Fallback: keep last directory
            preserved_dirs = (input_path.parent.name,)
    else:
        # No dataset found, keep last directory
        preserved_dirs = (input_path.parent.name,)

    # Build final path
    final_output_path = output_path / Path(*preserved_dirs) if preserved_dirs else output_path

    if mkdir:
        final_output_path.mkdir(parents=True, exist_ok=True)

    return final_output_path / input_path.name
