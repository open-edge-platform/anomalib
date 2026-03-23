# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Loss function for the PatchFlow Model Implementation.

PatchFlow uses a single normalizing flow on fused features, so the loss
operates on a single ``(B, C, H, W)`` hidden variable tensor and ``(B,)``
jacobians, normalized by the flattened feature dimension.
"""

import torch
from torch import nn


class PatchflowLoss(nn.Module):
    """PatchFlow Loss Module.

    Computes the negative log-likelihood loss normalized by the feature
    dimension of the fused hidden variable tensor.
    """

    @staticmethod
    def forward(hidden_variables: torch.Tensor, jacobians: torch.Tensor) -> torch.Tensor:
        """Calculate the PatchFlow loss.

        Args:
            hidden_variables (torch.Tensor): Hidden variable tensor of shape
                ``(B, C, H, W)`` from the normalizing flow.
            jacobians (torch.Tensor): Log determinant of the Jacobian of shape
                ``(B,)``.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        z_flat = hidden_variables.reshape(hidden_variables.shape[0], -1)
        return torch.mean(0.5 * torch.sum(z_flat**2, dim=1) - jacobians) / z_flat.shape[1]
