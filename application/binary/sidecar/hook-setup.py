# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime setup hook for environment configuration."""

import multiprocessing as mp
import os
import pathlib
import platform
import sys

# CRITICAL: Must call freeze_support() FIRST in runtime hook to prevent
# worker processes from executing this setup code in PyInstaller builds
mp.freeze_support()

system = platform.system()
print("Setup Hook: Detected operating system:", system)

# Set up paths for PyInstaller frozen app
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # Running in PyInstaller bundle
    bundle_dir = sys._MEIPASS
    print(f"Setup Hook: Running in PyInstaller bundle: {bundle_dir}")

    # Set alembic paths to point to bundled files
    alembic_ini = os.path.join(bundle_dir, "src", "alembic.ini")
    alembic_dir = os.path.join(bundle_dir, "src", "alembic")

    # Verify files exist
    if pathlib.Path(alembic_ini).exists():
        print(f"Setup Hook: Found alembic.ini at: {alembic_ini}")
        os.environ["ALEMBIC_CONFIG_PATH"] = alembic_ini
    else:
        print(f"Setup Hook: WARNING - alembic.ini not found at: {alembic_ini}")

    if pathlib.Path(alembic_dir).exists():
        print(f"Setup Hook: Found alembic directory at: {alembic_dir}")
        os.environ["ALEMBIC_SCRIPT_LOCATION"] = alembic_dir
    else:
        print(f"Setup Hook: WARNING - alembic directory not found at: {alembic_dir}")

if system == "Windows":
    local_app_data = os.getenv("LOCALAPPDATA")
    if not local_app_data:
        raise OSError("LOCALAPPDATA environment variable is not set.")

    import ctypes

    GetCurrentPackageFamilyName = ctypes.windll.kernel32.GetCurrentPackageFamilyName  # type: ignore[attr-defined]
    GetCurrentPackageFamilyName.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_wchar_p]
    GetCurrentPackageFamilyName.restype = ctypes.c_long

    length = ctypes.c_uint(256)
    package_family_name_buffer = ctypes.create_unicode_buffer(256)

    result = GetCurrentPackageFamilyName(ctypes.byref(length), package_family_name_buffer)
    if result == 0:
        package_family_name = package_family_name_buffer.value
        print("Setup Hook: Application runs in a UWP context. Package Family Name:", package_family_name)

        app_data_folder = os.path.join(local_app_data, "Packages", package_family_name, "LocalState")

        print("Setup Hook: Using local state folder:", app_data_folder)
        os.environ["DB_DATA_DIR"] = app_data_folder

        print("Setup Hook: Writing log to:", app_data_folder)
        os.environ["LOGS_DIR"] = app_data_folder
    else:
        print("Setup Hook: Application doesn't run in a UWP context; skipping folder setup.")
