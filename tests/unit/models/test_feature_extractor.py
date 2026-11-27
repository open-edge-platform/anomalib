# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test feature extractors."""

import pytest
import torch
from torchvision.models import ResNet18_Weights, resnet18

from anomalib.models.components.feature_extractors import (
    TimmFeatureExtractor,
    dryrun_find_featuremap_dims,
)


class TestFeatureExtractor:
    """Test the feature extractor."""

    @staticmethod
    @pytest.mark.parametrize("backbone", ["resnet18", "wide_resnet50_2"])
    @pytest.mark.parametrize("pretrained", [True, False])
    def test_timm_feature_extraction(backbone: str, pretrained: bool) -> None:
        """Test if the feature extractor can be instantiated and if the output is as expected."""
        layers = ["layer1", "layer2", "layer3"]
        model = TimmFeatureExtractor(backbone=backbone, layers=layers, pre_trained=pretrained)
        test_input = torch.rand((32, 3, 256, 256))
        features = model(test_input)

        if backbone == "resnet18":
            assert features["layer1"].shape == torch.Size((32, 64, 64, 64))
            assert features["layer2"].shape == torch.Size((32, 128, 32, 32))
            assert features["layer3"].shape == torch.Size((32, 256, 16, 16))
            assert model.out_dims == [64, 128, 256]
            assert model.idx == [1, 2, 3]
        elif backbone == "wide_resnet50_2":
            assert features["layer1"].shape == torch.Size((32, 256, 64, 64))
            assert features["layer2"].shape == torch.Size((32, 512, 32, 32))
            assert features["layer3"].shape == torch.Size((32, 1024, 16, 16))
            assert model.out_dims == [256, 512, 1024]
            assert model.idx == [1, 2, 3]

    @staticmethod
    def test_timm_feature_extraction_custom_backbone() -> None:
        """Test if the feature extractor can be instantiated and if the output is as expected."""
        layers = ["layer1", "layer2", "layer3"]
        backbone = resnet18(weights=ResNet18_Weights)
        model = TimmFeatureExtractor(backbone=backbone, layers=layers, pre_trained=False)
        test_input = torch.rand((32, 3, 256, 256))
        features = model(test_input)

        assert features["layer1"].shape == torch.Size((32, 64, 64, 64))
        assert features["layer2"].shape == torch.Size((32, 128, 32, 32))
        assert features["layer3"].shape == torch.Size((32, 256, 16, 16))
        assert model.out_dims == [64, 128, 256]

    @staticmethod
    @pytest.mark.parametrize("return_class_token", [False, True])
    def test_timm_feature_extraction_nlc_dinov2(return_class_token: bool) -> None:
        """Test the token (NLC) mode used for DINOv2 ViT backbones."""
        layers = ["blocks.2", "blocks.9"]
        model = TimmFeatureExtractor(
            backbone="vit_base_patch14_reg4_dinov2",
            layers=layers,
            pre_trained=False,
            output_fmt="NLC",
            return_class_token=return_class_token,
            norm=False,
            dynamic_img_size=True,
        )
        # 392 / 14 = 28 patches per side -> 784 patch tokens; reg4 model has 5 prefix tokens.
        features = model(torch.rand((2, 3, 392, 392)))
        num_patches = 28 * 28
        expected_tokens = num_patches + (model.num_prefix_tokens if return_class_token else 0)

        assert set(features) == set(layers)
        for layer in layers:
            assert features[layer].shape == torch.Size((2, expected_tokens, 768))
        assert model.idx == [2, 9]
        assert model.out_dims == [768, 768]
        assert model.patch_size == 14
        assert model.num_register_tokens == 4
        assert model.num_prefix_tokens == 5

    @staticmethod
    def test_timm_feature_extraction_invalid_output_fmt() -> None:
        """Test that an invalid output_fmt raises a ValueError."""
        with pytest.raises(ValueError, match="output_fmt must be one of"):
            TimmFeatureExtractor(backbone="resnet18", layers=["layer1"], output_fmt="invalid")


@pytest.mark.parametrize("backbone", ["resnet18", "wide_resnet50_2"])
@pytest.mark.parametrize("input_size", [(256, 256), (224, 224), (128, 128)])
def test_dryrun_find_featuremap_dims(backbone: str, input_size: tuple[int, int]) -> None:
    """Use the function and check the expected output format."""
    layers = ["layer1", "layer2", "layer3"]
    model = TimmFeatureExtractor(backbone=backbone, layers=layers, pre_trained=True)
    mapping = dryrun_find_featuremap_dims(model, input_size, layers)
    for lay in layers:
        layer_mapping = mapping[lay]
        num_features = layer_mapping["num_features"]
        assert isinstance(num_features, int), f"{type(num_features)}"
        resolution = layer_mapping["resolution"]
        assert isinstance(resolution, tuple), f"{type(resolution)}"
        assert len(resolution) == len(input_size), f"{len(resolution)}, {len(input_size)}"
        assert all(isinstance(x, int) for x in resolution), f"{[type(x) for x in resolution]}"
