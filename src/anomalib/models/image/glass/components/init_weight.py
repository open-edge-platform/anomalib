# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Initializes network weights using Xavier normal initialization."""

import torch
from torch import nn


def init_weight(m: nn.Module) -> None:
    """Initializes network weights using Xavier normal initialization.

    Applies Xavier initialization for linear layers and normal initialization
    for convolutional and batch normalization layers.
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    if isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
