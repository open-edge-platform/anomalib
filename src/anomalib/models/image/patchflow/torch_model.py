# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PatchFlow Torch Model Implementation.

PatchFlow fuses multi-scale backbone features into a single tensor, compresses
them with a 1x1 convolution adaptor, and runs a single normalizing flow for
anomaly detection.
"""

from collections.abc import Callable

import timm
import torch
from FrEIA.framework import SequenceINN
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.data import InferenceBatch
from anomalib.models.components.dinov2 import DinoV2Loader
from anomalib.models.components.flow import AllInOneBlock

from .anomaly_map import AnomalyMapGenerator

# DINOv2 model name prefix used to distinguish from timm CNN backbones.
_DINOV2_PREFIX = "dinov2"


def _build_subnet_constructor(hidden_dim: int) -> Callable:
    """Build a subnet constructor for the normalizing flow coupling blocks.

    Args:
        hidden_dim: Number of hidden channels in the subnet.

    Returns:
        Callable that creates a subnet given ``(in_channels, out_channels)``.
    """

    def subnet_conv(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    return subnet_conv


class PatchflowModel(nn.Module):
    """PatchFlow model for anomaly detection.

    Extracts multi-scale features from a frozen backbone, fuses them into a
    single tensor, adapts the channel dimension, and passes through a single
    normalizing flow.

    Args:
        input_size: Model input size ``(H, W)``.
        backbone: timm model name or DINOv2 model name (e.g.
            ``"tf_efficientnet_b5"`` or ``"dinov2_vit_base_14"``).
        pre_trained: Whether to use pre-trained backbone weights.
        flow_steps: Number of coupling blocks in the normalizing flow.
        flow_feature_dim: Channel dimension after the feature adaptor.
        num_scales: Number of input resolutions for multi-scale extraction.
        patch_size: Kernel size of the AvgPool for local aggregation.
        flow_hidden_dim: Hidden channels in the flow subnet.
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        backbone: str = "tf_efficientnet_b5",
        pre_trained: bool = True,
        flow_steps: int = 1,
        flow_feature_dim: int = 128,
        num_scales: int = 3,
        patch_size: int = 3,
        flow_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_scales = num_scales
        self.backbone_name = backbone
        self.is_dinov2 = backbone.startswith(_DINOV2_PREFIX)

        # --- Feature extractor (frozen) ---
        if self.is_dinov2:
            self.feature_extractor = DinoV2Loader.from_name(backbone)
            # Use last 3 intermediate layers by default
            self.dino_layer_indices: list[int] = [9, 10, 11]
            embed_dim: int = self.feature_extractor.embed_dim
            total_channels = embed_dim * len(self.dino_layer_indices) * num_scales
        else:
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=pre_trained,
                features_only=True,
                out_indices=[2, 3, 4],
            )
            layer_channels: list[int] = self.feature_extractor.feature_info.channels()
            total_channels = sum(layer_channels) * num_scales

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # --- Feature fuser (AvgPool + upsample + concat) ---
        self.avg_pool = nn.AvgPool2d(kernel_size=patch_size, stride=1, padding=patch_size // 2)
        self.fused_spatial_size = (input_size[0] // 8, input_size[1] // 8)

        # --- Feature adaptor (1x1 conv) ---
        self.feature_adaptor = nn.Conv2d(total_channels, flow_feature_dim, kernel_size=1)

        # --- Normalizing flow ---
        self.flow = SequenceINN(flow_feature_dim, *self.fused_spatial_size)
        for _ in range(flow_steps):
            self.flow.append(
                AllInOneBlock,
                subnet_constructor=_build_subnet_constructor(flow_hidden_dim),
            )

        # --- Anomaly map generator ---
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _extract_cnn_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale CNN features.

        Runs the backbone at ``num_scales`` resolutions (1x, 0.5x, 0.25x, ...)
        and returns all feature maps in a flat list.
        """
        all_features: list[torch.Tensor] = []
        for s in range(self.num_scales):
            if s > 0:
                scaled = F.interpolate(
                    x,
                    size=(self.input_size[0] // (2**s), self.input_size[1] // (2**s)),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                scaled = x
            features = self.feature_extractor(scaled)
            all_features.extend(features)
        return all_features

    def _extract_dinov2_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale DINOv2 features.

        Runs the DINOv2 backbone at ``num_scales`` resolutions and collects
        intermediate layer outputs reshaped to ``(B, C, H', W')``.
        """
        patch_size: int = self.feature_extractor.patch_size
        all_features: list[torch.Tensor] = []
        for s in range(self.num_scales):
            if s > 0:
                h = self.input_size[0] // (2**s)
                w = self.input_size[1] // (2**s)
                # Ensure dimensions are divisible by the DINOv2 patch size
                h = (h // patch_size) * patch_size
                w = (w // patch_size) * patch_size
                scaled = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
            else:
                scaled = x
            features = self.feature_extractor.get_intermediate_layers(
                scaled,
                n=self.dino_layer_indices,
                reshape=True,
            )
            all_features.extend(features)
        return all_features

    # ------------------------------------------------------------------
    # Feature fusion
    # ------------------------------------------------------------------

    def _fuse_features(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Fuse a list of feature maps into a single tensor.

        Applies local average pooling, upsamples to a common spatial size,
        and concatenates along the channel axis.
        """
        fused: list[torch.Tensor] = []
        for feat in features:
            feat = self.avg_pool(feat)
            feat = F.interpolate(feat, size=self.fused_spatial_size, mode="bilinear", align_corners=False)
            fused.append(feat)
        return torch.cat(fused, dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | InferenceBatch:
        """Forward pass.

        Args:
            input_tensor: Input image tensor of shape ``(B, 3, H, W)``.

        Returns:
            During training: ``(hidden_variables, log_jacobians)`` as single tensors.
            During inference: :class:`InferenceBatch` with ``pred_score`` and
                ``anomaly_map``.
        """
        self.feature_extractor.eval()

        # 1. Extract multi-scale features
        if self.is_dinov2:
            features = self._extract_dinov2_features(input_tensor)
        else:
            features = self._extract_cnn_features(input_tensor)

        # 2. Fuse features
        fused = self._fuse_features(features)

        # 3. Adapt channel dimension
        adapted = self.feature_adaptor(fused)

        # 4. Normalizing flow
        hidden_variables, log_jacobians = self.flow(adapted)

        if self.training:
            return hidden_variables, log_jacobians

        # 5. Generate anomaly map
        anomaly_map = self.anomaly_map_generator(hidden_variables)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
