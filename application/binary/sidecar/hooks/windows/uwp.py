# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime hook: UWP context detection and folder setup (Windows only)."""

import ctypes
import os
import shutil
from pathlib import Path

GetCurrentPackageFamilyName = ctypes.windll.kernel32.GetCurrentPackageFamilyName  # type: ignore[attr-defined]
GetCurrentPackageFamilyName.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_wchar_p]
GetCurrentPackageFamilyName.restype = ctypes.c_long

GetCurrentPackagePath = ctypes.windll.kernel32.GetCurrentPackagePath  # type: ignore[attr-defined]
GetCurrentPackagePath.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_wchar_p]
GetCurrentPackagePath.restype = ctypes.c_long


def _get_current_package_family_name() -> str | None:
    length = ctypes.c_uint(256)
    buffer = ctypes.create_unicode_buffer(256)

    result = GetCurrentPackageFamilyName(ctypes.byref(length), buffer)
    return buffer.value if result == 0 else None


def _get_current_package_path() -> str | None:
    length = ctypes.c_uint(256)
    buffer = ctypes.create_unicode_buffer(256)

    result = GetCurrentPackagePath(ctypes.byref(length), buffer)
    return buffer.value if result == 0 else None


def _copy_initial_data(app_data_folder: Path) -> None:
    package_path = _get_current_package_path()
    print(f"Setup Hook: Application package path: {package_path}")
    if not package_path:
        return
    initial_data_path = Path(package_path) / "InitialData"
    if not os.path.exists(initial_data_path):
        return
    for item in os.listdir(initial_data_path):
        destination_path = app_data_folder / item
        if os.path.exists(destination_path):
            continue
        print(f"Setup Hook: Copying initial data: {item}. Destination: {destination_path}")
        try:
            shutil.copytree(initial_data_path / item, destination_path)
        except OSError as e:
            print(f"Setup Hook: Failed to copy initial data {item}", e)


def _main() -> None:
    local_app_data = os.getenv("LOCALAPPDATA")
    if not local_app_data:
        raise OSError("LOCALAPPDATA environment variable is not set.")

    package_family_name = _get_current_package_family_name()
    if not package_family_name:
        # Not a UWP/MSIX package (e.g. MSI install or standalone exe).
        # Still need a writable location — Program Files is read-only.
        print("Setup Hook: Application doesn't run in a UWP context; using LOCALAPPDATA.")
        app_data_folder = Path(local_app_data) / "Anomalib Studio"
        app_data_folder.mkdir(parents=True, exist_ok=True)

        data_dir = str(app_data_folder / "data")
        logs_dir = str(app_data_folder / "logs")
        print(f"Setup Hook: Setting data directory to: {data_dir}")
        os.environ["DATA_DIR"] = data_dir
        print(f"Setup Hook: Setting logs directory to: {logs_dir}")
        os.environ["LOG_DIR"] = logs_dir
        return

    print(f"Setup Hook: Application runs in a UWP context. Package Family Name: {package_family_name}")

    app_data_folder = Path(local_app_data) / "Packages" / package_family_name / "LocalState"

    print(f"Setup Hook: Using local state folder: {app_data_folder}")
    os.environ["DATA_DIR"] = str(app_data_folder)

    print(f"Setup Hook: Writing log to: {app_data_folder}")
    os.environ["LOG_DIR"] = str(app_data_folder)

    _copy_initial_data(app_data_folder)


_main()
