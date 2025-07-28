#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Quick install (uses default PyTorch from PyPI)
pip install anomalib

# To ensure a specific hardware backend, you must specify an extra.
# For example, for CPU:
pip install "anomalib[cpu]"

# Or for CUDA 12.4:
pip install "anomalib[cu124]"

# For a full installation with all optional dependencies on CPU:
pip install "anomalib[full,cpu]"

# To install from source for development on CPU:
git clone https://github.com/open-edge-platform/anomalib.git
cd anomalib || exit

# Development installation on CPU
pip install -e ".[dev,cpu]"
