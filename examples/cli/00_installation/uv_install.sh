#!/bin/bash

# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Verify the pinned standalone installer before running it.
uv_installer=$(mktemp)
trap 'rm -f "$uv_installer"' EXIT
curl -LsSf https://github.com/astral-sh/uv/releases/download/0.13.7/uv-installer.sh \
    -o "$uv_installer"
uv_installer_sha256=d263ee68314cf5bea36b08e6bd9e2c4563240d7adb73756cf33af5c199b92db4
if command -v sha256sum >/dev/null 2>&1; then
    printf '%s  %s\n' "$uv_installer_sha256" "$uv_installer" | sha256sum -c -
else
    printf '%s  %s\n' "$uv_installer_sha256" "$uv_installer" | shasum -a 256 -c -
fi
sh "$uv_installer"
rm -f "$uv_installer"
trap - EXIT

# Quick install (uses default PyTorch from PyPI)
uv pip install anomalib

# To ensure a specific hardware backend, use one of the following extras:

# CPU support (works on all platforms)
uv pip install "anomalib[cpu]"

# CUDA support (Linux/Windows with NVIDIA GPU)
uv pip install "anomalib[cu126]"  # CUDA 12.6
uv pip install "anomalib[cu130]"  # CUDA 13.0

# ROCm support (Linux with AMD GPU)
uv pip install "anomalib[rocm]"

# Intel XPU support (Linux/Windows with Intel GPU)
uv pip install "anomalib[xpu]"

# You can combine extras. For example, to install with CUDA 13.0 and OpenVINO support:
uv pip install "anomalib[openvino,cu130]"

# For a full installation with all optional dependencies on CPU:
uv pip install "anomalib[full,cpu]"

# To install from source for development:
git clone https://github.com/open-edge-platform/anomalib.git
cd anomalib

# Create a virtual environment and sync with the lockfile for a specific backend:
uv venv
uv sync --extra cpu

# Or for CUDA 13.0
uv sync --extra cu130

# For a full development environment on CPU
uv sync --extra dev --extra cpu
