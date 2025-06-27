#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Quick install (uses default PyTorch from PyPI)
uv pip install anomalib

# To ensure a specific hardware backend, use one of the following extras:

# CPU support (works on all platforms)
uv pip install "anomalib[cpu]"

# CUDA support (Linux/Windows with NVIDIA GPU)
uv pip install "anomalib[cu124]"  # CUDA 12.4
uv pip install "anomalib[cu121]"  # CUDA 12.1
uv pip install "anomalib[cu118]"  # CUDA 11.8

# ROCm support (Linux with AMD GPU)
uv pip install "anomalib[rocm]"

# Intel XPU support (Linux/Windows with Intel GPU)
uv pip install "anomalib[xpu]"

# You can combine extras. For example, to install with CUDA 12.4 and OpenVINO support:
uv pip install "anomalib[openvino,cu124]"

# For a full installation with all optional dependencies on CPU:
uv pip install "anomalib[full,cpu]"

# To install from source for development:
git clone https://github.com/open-edge-platform/anomalib.git
cd anomalib

# Create a virtual environment and sync with the lockfile for a specific backend:
uv venv
uv sync --extra cpu

# Or for CUDA 12.4
uv sync --extra cu124

# For a full development environment on CPU
uv sync --extra dev --extra cpu
