# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GeneralAD model."""

from .lightning_model import GeneralAD
from .torch_model import GeneralADModel

__all__ = ["GeneralAD", "GeneralADModel"]
