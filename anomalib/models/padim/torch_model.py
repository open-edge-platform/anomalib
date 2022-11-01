"""PyTorch model for the PaDiM model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from random import sample
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from anomalib.models.components import FeatureExtractor, MultiVariateGaussian
from anomalib.models.padim.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler

_DIMS = {
    "resnet18": {"orig_dims": 448, "emb_scale": 4},
    "wide_resnet50_2": {"orig_dims": 1792, "emb_scale": 4},
}

# defaults from the paper
_N_FEATURES_DEFAULTS = {
    "resnet18": 100,
    "wide_resnet50_2": 550,
}


def _deduce_dims(
    feature_extractor: FeatureExtractor, input_size: Tuple[int, int], layers: List[str]
) -> Tuple[int, int]:
    """Run a dry run to deduce the dimensions of the extracted features.

    Returns:
        Tuple[int, int]: Dimensions of the extracted features: (n_dims_original, n_patches)
    """

    dryrun_input = torch.empty(1, 3, *input_size)
    dryrun_features = feature_extractor(dryrun_input)

    # the first layer in `layers` is the largest spatial size
    dryrun_emb_first_layer = dryrun_features[layers[0]]
    n_patches = torch.tensor(dryrun_emb_first_layer.shape[-2:]).prod().int().item()

    # the original embedding size is the sum of the channels of all layers
    n_features_original = sum(dryrun_features[layer].shape[1] for layer in layers)

    return n_features_original, n_patches


class PadimModel(nn.Module):
    """Padim Module.

    Args:
        input_size (Tuple[int, int]): Input size for the model.
        layers (List[str]): Layers used for feature extraction
        backbone (str, optional): Pre-trained model backbone. Defaults to "resnet18".
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        layers: List[str],
        backbone: str = "resnet18",
        pre_trained: bool = True,
        n_features: Optional[int] = None,
    ):
        super().__init__()
        self.tiler: Optional[Tiler] = None

        self.backbone = backbone
        self.layers = layers
        self.feature_extractor = FeatureExtractor(backbone=self.backbone, layers=layers, pre_trained=pre_trained)

        if backbone in _DIMS:
            backbone_dims = _DIMS[backbone]
            self.n_features_original = backbone_dims["orig_dims"]
            emb_scale = backbone_dims["emb_scale"]
            patches_dims = torch.tensor(input_size) / emb_scale
            self.n_patches = patches_dims.ceil().prod().int().item()

        else:
            self.n_features_original, self.n_patches = _deduce_dims(self.feature_extractor, input_size, self.layers)

        if n_features is None:

            if self.backbone in _N_FEATURES_DEFAULTS:
                n_features = _N_FEATURES_DEFAULTS[self.backbone]

            else:
                raise ValueError(
                    f"{self.__class__.__name__}.n_features must be specified for backbone {self.backbone}. "
                    f"Default values are available for: {sorted(_N_FEATURES_DEFAULTS.keys())}"
                )
        assert (
            n_features <= self.n_features_original
        ), f"n_features ({n_features}) must be <= n_features_original ({self.n_features_original})"
        self.n_features = n_features
        # pylint: disable=not-callable
        # Since idx is randomly selected, save it with model to get same results
        self.register_buffer(
            "idx",
            torch.tensor(sample(range(0, self.n_features_original), self.n_features)),
        )
        self.idx: Tensor
        self.loss = None
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=input_size)

        self.gaussian = MultiVariateGaussian(self.n_features, self.n_patches)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Forward-pass image-batch (N, C, H, W) into model to extract features.

        Args:
            input_tensor: Image-batch (N, C, H, W)
            input_tensor: Tensor:

        Returns:
            Features from single/multiple layers.

        Example:
            >>> x = torch.randn(32, 3, 224, 224)
            >>> features = self.extract_features(input_tensor)
            >>> features.keys()
            dict_keys(['layer1', 'layer2', 'layer3'])

            >>> [v.shape for v in features.values()]
            [torch.Size([32, 64, 56, 56]),
            torch.Size([32, 128, 28, 28]),
            torch.Size([32, 256, 14, 14])]
        """

        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            embeddings = self.generate_embedding(features)

        if self.tiler:
            embeddings = self.tiler.untile(embeddings)

        if self.training:
            output = embeddings
        else:
            output = self.anomaly_map_generator(
                embedding=embeddings, mean=self.gaussian.mean, inv_covariance=self.gaussian.inv_covariance
            )
        return output

    def generate_embedding(self, features: Dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features (Dict[str, Tensor]): Hierarchical feature map from a CNN (ResNet18 or WideResnet)

        Returns:
            Embedding vector
        """

        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        # subsample embeddings
        idx = self.idx.to(embeddings.device)
        embeddings = torch.index_select(embeddings, 1, idx)
        return embeddings
