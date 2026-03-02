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
    alembic_ini = os.path.join(bundle_dir, "alembic.ini")
    alembic_dir = os.path.join(bundle_dir, "alembic")

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

if system in ["Linux", "Darwin"]:
    home_dir = os.path.expanduser("~")
    app_data_dir = os.path.join(home_dir, ".anomalib_studio")

    data_dir = os.path.join(app_data_dir, "data")
    print("Setup Hook: Setting data directory to:", data_dir)
    os.environ["DATA_DIR"] = data_dir

    logs_dir = os.path.join(app_data_dir, "logs")
    print("Setup Hook: Setting logs directory to:", logs_dir)
    os.environ["LOG_DIR"] = logs_dir
