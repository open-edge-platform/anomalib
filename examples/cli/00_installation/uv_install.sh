#!/bin/bash

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Basic installation from PyPI
uv add anomalib

# Full installation with all dependencies
uv add anomalib[full]

# Install from source for development
git clone https://github.com/open-edge-platform/anomalib.git
cd anomalib

# Install in development mode
uv sync

# Full development installation with all dependencies
uv sync --extra dev
