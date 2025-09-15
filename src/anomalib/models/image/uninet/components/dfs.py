"""Domain Related Feature Selection."""

# Original Code
# Copyright (c) 2025 Shun Wei
# https://github.com/pangdatangtt/UniNet
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import functional


class DomainRelatedFeatureSelection(nn.Module):
    """Domain Related Feature Selection.

    It is used to select the domain-related features from the source and target features.

    Args:
        num_channels (int): Number of channels in the features. Defaults to 256.
        learnable (bool): Whether to use learnable theta. Theta controls the domain-related feature selection.
    """

    def __init__(self, num_channels: int = 256, learnable: bool = True) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.theta1 = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.theta2 = nn.Parameter(torch.zeros(1, num_channels * 2, 1, 1))
        self.theta3 = nn.Parameter(torch.zeros(1, num_channels * 4, 1, 1))
        self.learnable = learnable

    def _get_theta(self, idx: int) -> torch.Tensor:
        match idx:
            case 1:
                return self.theta1
            case 2:
                return self.theta2
            case 3:
                return self.theta3
            case _:
                msg = f"Invalid index: {idx}"
                raise ValueError(msg)

    def forward(
        self,
        source_features: list[torch.Tensor],
        target_features: list[torch.Tensor],
        conv: bool = False,
        maximize: bool = True,
    ) -> list[torch.Tensor]:
        """Domain related feature selection.

        Args:
            source_features (list[torch.Tensor]): Source features.
            target_features (list[torch.Tensor]): Target features.
            conv (bool): Whether to use convolutional domain-related feature selection.
                Defaults to False.
            maximize (bool): Used for weights computation. If True, the weights are computed by subtracting the
                max value from the target feature. Defaults to True.

        Returns:
            list[torch.Tensor]: Domain related features.
        """
        features = []
        for idx, (source_feature, target_feature) in enumerate(zip(source_features, target_features, strict=True)):
            theta = 1
            if self.learnable:
                #  to avoid losing local weight, theta should be as non-zero value as possible
                if idx < 3:
                    theta = torch.clamp(torch.sigmoid(self._get_theta(idx + 1)) * 1.0 + 0.5, max=1)
                else:
                    theta = torch.clamp(torch.sigmoid(self._get_theta(idx - 2)) * 1.0 + 0.5, max=1)

            b, c, h, w = source_feature.shape
            if not conv:
                prior_flat = target_feature.view(b, c, -1)
                if maximize:
                    prior_flat_ = prior_flat.max(dim=-1, keepdim=True)[0]
                    prior_flat = prior_flat - prior_flat_
                weights = functional.softmax(prior_flat, dim=-1)
                weights = weights.view(b, c, h, w)

                global_inf = target_feature.mean(dim=(-2, -1), keepdim=True)

                inter_weights = weights * (theta + global_inf)

                x_ = source_feature * inter_weights
                features.append(x_)

        return features
