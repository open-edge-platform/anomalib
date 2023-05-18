"""Path Utils."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from pathlib import Path, PureWindowsPath

import pandas as pd
from torchvision.datasets.folder import IMG_EXTENSIONS


class DirType(str, Enum):
    """Dir type names."""

    NORMAL = "normal"
    ABNORMAL = "abnormal"
    NORMAL_TEST = "normal_test"
    NORMAL_DEPTH = "normal_depth"
    ABNORMAL_DEPTH = "abnormal_depth"
    NORMAL_TEST_DEPTH = "normal_test_depth"
    MASK = "mask_dir"


def _check_and_convert_path(path: str | Path) -> Path:
    """Check an input path, and convert to Pathlib object.

    Args:
        path (str | Path): Input path.

    Returns:
        Path: Output path converted to pathlib object.
    """
    if not isinstance(path, Path):
        path = Path(path)
    return path


def _prepare_files_labels(
    path: str | Path, path_type: str, extensions: tuple[str, ...] | None = None
) -> tuple[list, list]:
    """Return a list of filenames and list corresponding labels.

    Args:
        path (str | Path): Path to the directory containing images.
        path_type (str): Type of images in the provided path ("normal", "abnormal", "normal_test")
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the
            directory.

    Returns:
        List, List: Filenames of the images provided in the paths, labels of the images provided in the paths
    """
    path = _check_and_convert_path(path)
    if extensions is None:
        extensions = IMG_EXTENSIONS

    if isinstance(extensions, str):
        extensions = (extensions,)

    filenames = [f for f in path.glob(r"**/*") if f.suffix in extensions and not f.is_dir()]
    if not filenames:
        raise RuntimeError(f"Found 0 {path_type} images in {path}")

    labels = [path_type] * len(filenames)

    return filenames, labels


def _prepare_filemeta_from_csv(path: str | Path) -> pd.DataFrame:
    """Return a DataFrame of dataset file metadata from CSV file

    Args:
        path (str | Path): Path to the CSV file
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the
            CSV file.

    Returns:
        pd.DataFrame: Contents of CSV dataset file, with at least `image_path` and `label` columns
    """
    path = _check_and_convert_path(path)

    csv_data = pd.read_csv(path)
    if len(csv_data) == 0:
        raise RuntimeError(f"Empty CSV file in {path}")

    if "image_path" not in csv_data or "label" not in csv_data:
        raise RuntimeError(f"Invalid CSV file (missing required columns) {path}")

    # Convert to posix path for best compatibility
    csv_data["image_path"] = csv_data.apply(lambda row: Path(PureWindowsPath(row.image_path).as_posix()), axis=1)

    if len(csv_data) == 0:
        raise RuntimeError(f"Found 0 images in CSV file in {path}")

    # Any extra columns will be returned to caller
    return csv_data


def _resolve_path(folder: str | Path, root: str | Path | None = None) -> Path:
    """Combines root and folder and returns the absolute path.

    This allows users to pass either a root directory and relative paths, or absolute paths to each of the
    image sources. This function makes sure that the samples dataframe always contains absolute paths.

    Args:
        folder (str | Path | None): Folder location containing image or mask data.
        root (str | Path | None): Root directory for the dataset.
    """
    folder = Path(folder)
    if folder.is_absolute():
        # path is absolute; return unmodified
        path = folder
    # path is relative.
    elif root is None:
        # no root provided; return absolute path
        path = folder.resolve()
    else:
        # root provided; prepend root and return absolute path
        path = (Path(root) / folder).resolve()
    return path
