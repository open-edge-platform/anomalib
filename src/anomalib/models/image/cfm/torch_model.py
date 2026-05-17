# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model implementation for CFM."""

from __future__ import annotations

import torch
from torch import nn

from anomalib.data import InferenceBatch

from .anomaly_map import CFMAnomalyMapGenerator
from .components import FeatureProjectionMLP, MultimodalFeatures


class CFMModel(nn.Module):
    """Crossmodal Feature Mapping (CFM) model.

    Model learns from the correspondance between geometry(3D) and appearance (2D).
    """

    def __init__(
        self,
        rgb_backbone: str = "vit_base_patch8_224.dino",
        group_size: int = 128,
        num_group: int = 1024,
    ) -> None:
        """Initialise the multimodal mapping system.

        Args:
            rgb_backbone: Name of DINO model to upload.
            group_size: Dimension of groups for the Point Cloud (KNN).
            num_group: Number of groups (FPS) for the Point Cloud.
        """
        super().__init__()

        self.feature_extractors = MultimodalFeatures(
            rgb_backbone_name=rgb_backbone,
            group_size=group_size,
            num_group=num_group,
        ).eval()

        # Blocking the gradients: extractors aren't updated
        for param in self.feature_extractors.parameters():
            param.requires_grad = False

        # Dimensions of the features (DINO Base = 768, Point-MAE = 384)
        rgb_dim = 768
        xyz_dim = 384 * 3

        # MLP models that learn to predict one mode from another.
        self.xyz_to_rgb_mapper = FeatureProjectionMLP(in_features=xyz_dim, out_features=rgb_dim)
        self.rgb_to_xyz_mapper = FeatureProjectionMLP(in_features=rgb_dim, out_features=xyz_dim)

        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
        # This class manages the KNN blur and the spatial computation of anomalies
        self.anomaly_map_generator = CFMAnomalyMapGenerator()

    def mapper_parameters(self) -> list[torch.nn.Parameter]:
        """It returns only the mapping network parameters for the optimiser."""
        return list(self.xyz_to_rgb_mapper.parameters()) + list(self.rgb_to_xyz_mapper.parameters())

    @torch.no_grad()
    def extract_features(self, rgb: torch.Tensor, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the real features from the sensors(2D e 3D)."""
        self.feature_extractors.eval()
        rgb_feat, xyz_feat = self.feature_extractors.get_features_maps(rgb, xyz)
        return rgb_feat, xyz_feat

    def map_features(self, rgb_feat: torch.Tensor, xyz_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Crossmodal mapping."""
        pred_rgb = self.xyz_to_rgb_mapper(xyz_feat)
        pred_xyz = self.rgb_to_xyz_mapper(rgb_feat)
        return pred_rgb, pred_xyz

    def compute_losses(
        self,
        rgb_feat: torch.Tensor,
        xyz_feat: torch.Tensor,
        pred_rgb: torch.Tensor,
        pred_xyz: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluates the loss based on discrepancy between real and mapped features."""
        loss_rgb = 1 - self.cos_sim(pred_rgb, rgb_feat).mean()
        loss_xyz = 1 - self.cos_sim(pred_xyz, xyz_feat).mean()

        return {
            "loss": loss_rgb + loss_xyz,
            "loss_rgb": loss_rgb,
            "loss_xyz": loss_xyz,
        }

    def _training_forward(self, rgb: torch.Tensor, xyz: torch.Tensor) -> dict[str, torch.Tensor]:
        """Step during training (normal samples only)."""
        rgb_feat, xyz_feat = self.extract_features(rgb, xyz)
        pred_rgb, pred_xyz = self.map_features(rgb_feat, xyz_feat)
        return self.compute_losses(rgb_feat, xyz_feat, pred_rgb, pred_xyz)

    def forward(self, rgb: torch.Tensor, xyz: torch.Tensor) -> dict[str, torch.Tensor] | InferenceBatch:
        """It manages the data flow based on the model's status."""
        if self.training:
            return self._training_forward(rgb, xyz)

        # Inference
        rgb_feat, xyz_feat = self.extract_features(rgb, xyz)
        pred_rgb, pred_xyz = self.map_features(rgb_feat, xyz_feat)

        # Generates the final map based on the difference between 'real' and 'mapped'
        anomaly_map, pred_score = self.anomaly_map_generator(
            rgb_feat=rgb_feat,
            xyz_feat=xyz_feat,
            pred_rgb=pred_rgb,
            pred_xyz=pred_xyz,
        )

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
