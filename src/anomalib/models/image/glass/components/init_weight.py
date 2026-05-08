# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Xavier/Kaiming weight initialization for GLASS network components."""

from torch import nn


def init_weight(module: nn.Module) -> None:
    """Initialize network weights using Xavier normal initialization.

    Applies Xavier initialization for linear layers, normal initialization
    for convolutional layers, and standard initialization for batch
    normalization layers (both 1D and 2D).

    Args:
        module (nn.Module): The module whose weights should be initialized.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)
    elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
