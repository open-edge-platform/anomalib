# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Maps input features to a fixed dimension using adaptive average pooling."""

import torch
import torch.nn.functional as f


class MeanMapper(torch.nn.Module):
    """Maps input features to a fixed dimension using adaptive average pooling.

    Input: Variable-sized feature tensors
    Output: Fixed-size feature representations
    """

    def __init__(self, preprocessing_dim: int) -> None:
        super().__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Applies adaptive average pooling to reshape features to a fixed size.

        Args:
            features (torch.Tensor): Input tensor of shape (B, *) where * denotes
            any number of remaining dimensions. It is flattened before pooling.

        Returns:
            torch.Tensor: Output tensor of shape (B, D), where D is `preprocessing_dim`.
        """
        features = features.reshape(len(features), 1, -1)
        return f.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Preprocessing(torch.nn.Module):
    """Handles initial feature preprocessing across multiple input dimensions.

    Input: List of features from different backbone layers
    Output: Processed features with consistent dimensionality
    """

    def __init__(self, input_dims: list[int | tuple[int, int]], output_dim: int) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for _ in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Applies preprocessing modules to a list of input feature tensors.

        Args:
            features (list of torch.Tensor): List of feature maps from different
                layers of the backbone network. Each tensor can have a different shape.

        Returns:
            torch.Tensor: A single tensor with shape (B, N, D), where B is the batch size,
            N is the number of feature maps, and D is the output dimension (`output_dim`).
        """
        features_ = []
        for module, feature in zip(self.preprocessing_modules, features, strict=False):
            features_.append(module(feature))
        return torch.stack(features_, dim=1)
