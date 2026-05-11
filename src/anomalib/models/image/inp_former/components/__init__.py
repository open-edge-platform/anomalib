# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Components module for INP-Former model.

This module provides all the necessary components for the INP-Former
architecture.
"""

# Layer components
from .layers import AggregationBlock, PrototypeBlock

__all__ = [
    # Layers
    "AggregationBlock",
    "PrototypeBlock",
]
