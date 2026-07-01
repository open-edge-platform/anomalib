# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PatchFlow Torch Model Implementation.

PatchFlow fuses multi-scale backbone features into a single tensor, compresses
them with a 1x1 convolution adaptor, and runs a single normalizing flow for
anomaly detection.
"""

from collections.abc import Callable

import torch
from FrEIA.framework import SequenceINN
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.data import InferenceBatch
from anomalib.models.components.feature_extractors import TimmFeatureExtractor
from anomalib.models.components.flow import AllInOneBlock

from .anomaly_map import AnomalyMapGenerator

# DINOv2 model name marker used to distinguish from timm CNN backbones.
# timm DINOv2 names contain "dinov2" (e.g. "vit_base_patch14_dinov2").
_DINOV2_PREFIX = "dinov2"

# Number of transformer blocks per DINOv2 ViT architecture.
_DINOV2_DEPTHS = {"small": 12, "base": 12, "large": 24, "giant": 40}


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
            ``"tf_efficientnet_b5"`` or ``"vit_base_patch14_dinov2"``).
        pre_trained: Whether to use pre-trained backbone weights.
        flow_steps: Number of coupling blocks in the normalizing flow.
        flow_feature_dim: Channel dimension after the feature adaptor.
        num_scales: Number of input resolutions for multi-scale extraction.
        patch_size: Kernel size of the AvgPool for local aggregation.
        flow_hidden_dim: Hidden channels in the flow subnet.
        crop_size: Optional center crop size ``(H, W)`` applied before
            feature extraction. The anomaly map is padded with ``-1`` back to
            ``input_size``. Defaults to ``None`` (no cropping).
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
        crop_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        if crop_size is not None and (crop_size[0] > input_size[0] or crop_size[1] > input_size[1]):
            msg = f"crop_size {crop_size} exceeds input_size {input_size}."
            raise ValueError(msg)
        self.crop_size = crop_size
        # Internal size used for backbone, flow, and anomaly map generation
        self._internal_size = crop_size if crop_size is not None else input_size
        self.num_scales = num_scales
        self.backbone_name = backbone
        self.is_dinov2 = _DINOV2_PREFIX in backbone

        # --- Feature extractor (frozen) ---
        if self.is_dinov2:
            # Determine depth from the architecture name to pick early/middle/late layers.
            try:
                num_blocks = next(depth for arch, depth in _DINOV2_DEPTHS.items() if arch in backbone)
            except StopIteration as exc:
                msg = f"Could not infer DINOv2 architecture from '{backbone}'. Expected one of {list(_DINOV2_DEPTHS)}."
                raise ValueError(msg) from exc
            self.dino_layer_indices: list[int] = [0, num_blocks // 2, num_blocks - 1]
            # Token-mode extractor (forward_intermediates) returning patch tokens per block.
            self.feature_extractor = TimmFeatureExtractor(
                backbone=backbone,
                layers=[f"blocks.{i}" for i in self.dino_layer_indices],
                pre_trained=pre_trained,
                requires_grad=False,
                output_fmt="NLC",
                return_class_token=False,
                norm=True,
                dynamic_img_size=True,
            )
            self._dino_patch_size = self.feature_extractor.patch_size

            # Ensure internal size is divisible by DINOv2 patch size
            dino_patch = self._dino_patch_size
            self._internal_size = (
                (self._internal_size[0] // dino_patch) * dino_patch,
                (self._internal_size[1] // dino_patch) * dino_patch,
            )
            # Enable cropping when crop_size was set, or when input_size
            # is not divisible by the DINOv2 patch size.
            if crop_size is not None or self._internal_size != input_size:
                self.crop_size = self._internal_size
            embed_dim = self.feature_extractor.out_dims[0]
            total_channels = embed_dim * len(self.dino_layer_indices) * num_scales
        else:
            self.feature_extractor = TimmFeatureExtractor(
                backbone=backbone,
                layers=["blocks.2", "blocks.4", "blocks.6"],
                pre_trained=pre_trained,
                requires_grad=False,
                output_fmt="NCHW",
            )
            total_channels = sum(self.feature_extractor.out_dims) * num_scales

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # --- Feature fuser (AvgPool + upsample + concat) ---
        self.avg_pool = nn.AvgPool2d(kernel_size=patch_size, stride=1, padding=patch_size // 2)
        if self.is_dinov2:
            dino_patch = self._dino_patch_size
            self.fused_spatial_size = (self._internal_size[0] // dino_patch, self._internal_size[1] // dino_patch)
        else:
            finest_stride = min(self.feature_extractor.reductions)
            self.fused_spatial_size = (
                self._internal_size[0] // finest_stride,
                self._internal_size[1] // finest_stride,
            )

        # --- Feature adaptor (1x1 conv) ---
        self.feature_adaptor = nn.Conv2d(total_channels, flow_feature_dim, kernel_size=1)

        # --- Normalizing flow ---
        self.flow = SequenceINN(flow_feature_dim, *self.fused_spatial_size)
        for _ in range(flow_steps):
            self.flow.append(
                AllInOneBlock,
                subnet_constructor=_build_subnet_constructor(flow_hidden_dim),
            )

        # --- Anomaly map generator (operates at crop_size, padded to input_size later) ---
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=self._internal_size)

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
                    size=(self._internal_size[0] // (2**s), self._internal_size[1] // (2**s)),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                scaled = x
            features = self.feature_extractor(scaled)
            all_features.extend(features.values())
        return all_features

    def _extract_dinov2_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale DINOv2 features.

        Runs the DINOv2 backbone at ``num_scales`` resolutions and collects
        intermediate layer outputs reshaped to ``(B, C, H', W')``.
        """
        patch_size: int = self._dino_patch_size
        all_features: list[torch.Tensor] = []
        for s in range(self.num_scales):
            if s > 0:
                h = self._internal_size[0] // (2**s)
                w = self._internal_size[1] // (2**s)
                # Ensure dimensions are divisible by the DINOv2 patch size
                h = (h // patch_size) * patch_size
                w = (w // patch_size) * patch_size
                scaled = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
            else:
                scaled = x
            # The extractor returns norm-applied patch tokens (B, N, D) per selected block;
            # reshape each to (B, D, H', W') maps for fusion (prefix tokens already stripped).
            hp = scaled.shape[-2] // patch_size
            wp = scaled.shape[-1] // patch_size
            for tokens in self.feature_extractor(scaled).values():
                b, _, d = tokens.shape
                all_features.append(tokens.reshape(b, hp, wp, d).permute(0, 3, 1, 2).contiguous())
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
            pooled = self.avg_pool(feat)
            pooled = F.interpolate(pooled, size=self.fused_spatial_size, mode="bilinear", align_corners=False)
            fused.append(pooled)
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

        # 0. Center crop if crop_size is set
        x = input_tensor
        if self.crop_size is not None:
            _, _, h_in, w_in = x.shape
            ch, cw = self.crop_size
            top = (h_in - ch) // 2
            left = (w_in - cw) // 2
            x = x[:, :, top : top + ch, left : left + cw]

        # 1. Extract multi-scale features (backbone is frozen, so disable grad to save memory/time)
        with torch.no_grad():
            features = self._extract_dinov2_features(x) if self.is_dinov2 else self._extract_cnn_features(x)

        # 2. Fuse features
        fused = self._fuse_features(features)

        # 3. Adapt channel dimension
        adapted = self.feature_adaptor(fused)

        # 4. Normalizing flow
        hidden_variables, log_jacobians = self.flow(adapted)

        if self.training:
            return hidden_variables, log_jacobians

        # 5. Generate anomaly map (at crop_size)
        anomaly_map = self.anomaly_map_generator(hidden_variables)

        # 6. Compute pred_score from cropped region before padding
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))

        # 7. Pad or crop anomaly map to match input_size
        _, _, h_in, w_in = input_tensor.shape
        _, _, h_am, w_am = anomaly_map.shape
        if (h_am, w_am) != (h_in, w_in):
            if h_am <= h_in and w_am <= w_in:
                # Pad smaller anomaly map to input size
                pad_top = (h_in - h_am) // 2
                pad_bottom = h_in - h_am - pad_top
                pad_left = (w_in - w_am) // 2
                pad_right = w_in - w_am - pad_left
                anomaly_map = F.pad(
                    anomaly_map,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=-1,
                )
            else:
                # Center-crop larger anomaly map to input size
                crop_top = (h_am - h_in) // 2
                crop_left = (w_am - w_in) // 2
                anomaly_map = anomaly_map[:, :, crop_top : crop_top + h_in, crop_left : crop_left + w_in]

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
