# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Components module for INP-Former model.

This module provides all the necessary components for the INP-Former
architecture including layers.
"""

# Layer components
from .layers import Aggregation_Block, Prototype_Block


__all__ = [
    # Layers
    "Aggregation_Block",
    "Prototype_Block",

]