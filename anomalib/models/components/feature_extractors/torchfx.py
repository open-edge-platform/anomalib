"""Feature Extractor based on TrochFX."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import List, Optional

from torch.fx.graph_module import GraphModule
from torchvision.models._api import WeightsEnum
from torchvision.models.feature_extraction import create_feature_extractor


def get_torchfx_feature_extractor(
    backbone: str, return_nodes: List[str], weights: Optional[WeightsEnum] = None
) -> GraphModule:
    """Extract features from a CNN.

    Args:
        backbone (nn.Module): The backbone to which the feature extraction hooks are attached.
        return_nodes (Iterable[str]): List of layer names of the backbone to which the hooks are attached.
            You can find the names of these nodes by using `get_graph_node_names` function.
        weights (Optional[WeightsEnum]): Weights enum to use for the model.
            These enums are defined in `torchvision.models.<model>`.

    Example:
        >>> import torch
        >>> from anomalib.models.components.feature_extractors import get_torchfx_feature_extractor
        >>> from torchvision.models.efficientnet import EfficientNet_B5_Weights

        >>> feature_extractor = get_torchfx_feature_extractor(
                backbone="efficientnet_b5", return_nodes=["6.8"], weights=EfficientNet_B5_Weights.DEFAULT
            )
        >>> input = torch.rand((32, 3, 256, 256))
        >>> features = feature_extractor(input)

        >>> [layer for layer in features.keys()]
            ["6.8"]
        >>> [feature.shape for feature in features.values()]
            [torch.Size([32, 304, 8, 8])]
    """
    try:
        models = importlib.import_module("torchvision.models")
        backbone_model = getattr(models, backbone)
    except ModuleNotFoundError as exception:
        raise ModuleNotFoundError(f"Backbone {backbone} not found in torchvision.models") from exception

    feature_extractor = create_feature_extractor(backbone_model(weights=weights).features, return_nodes).eval()
    for param in feature_extractor.parameters():
        param.requires_grad_(False)
    return feature_extractor
