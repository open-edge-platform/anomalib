"""PyTorch model for AI-VAD model implementation.

Paper https://arxiv.org/pdf/2212.00789.pdf
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import nn

from anomalib.models.ai_vad.density import CombinedDensityEstimator
from anomalib.models.ai_vad.features import FeatureExtractor
from anomalib.models.ai_vad.flow import FlowExtractor
from anomalib.models.ai_vad.regions import RegionExtractor


class AiVadModel(nn.Module):
    def __init__(
        self,
        # region-extraction params
        box_score_thresh: float = 0.8,
        # feature-extraction params
        n_velocity_bins: int = 8,
        use_velocity_features: bool = True,
        use_pose_features: bool = True,
        use_deep_features: bool = True,
        # density-estimation params
        n_components_velocity: int = 5,
        n_neighbors_pose: int = 1,
        n_neighbors_deep: int = 1,
    ):
        super().__init__()
        if not any((use_velocity_features, use_pose_features, use_deep_features)):
            raise ValueError("Select at least one feature type.")

        # initialize flow extractor
        self.flow_extractor = FlowExtractor()
        # initialize region extractor
        self.region_extractor = RegionExtractor(box_score_thresh=box_score_thresh)
        # initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            n_velocity_bins=n_velocity_bins,
            use_velocity_features=use_velocity_features,
            use_pose_features=use_pose_features,
            use_deep_features=use_deep_features,
        )
        # initialize density estimator
        self.density_estimator = CombinedDensityEstimator(
            use_velocity_features=use_velocity_features,
            use_pose_features=use_pose_features,
            use_deep_features=use_deep_features,
            n_components_velocity=n_components_velocity,
            n_neighbors_pose=n_neighbors_pose,
            n_neighbors_deep=n_neighbors_deep,
        )

    def forward(self, batch):
        self.flow_extractor.eval()
        self.region_extractor.eval()
        self.feature_extractor.eval()

        # 1. get first and last frame from clip
        first_frame = batch[:, 0, ...]
        last_frame = batch[:, -1, ...]

        # 2. extract flows and regions
        with torch.no_grad():
            flows = self.flow_extractor(first_frame, last_frame)
            regions = self.region_extractor(last_frame)

        # 3. extract pose, appearance and velocity features
        features_per_batch = self.feature_extractor(first_frame, flows, regions)

        if self.training:
            return features_per_batch

        # 4. estimate density
        box_scores = []
        image_scores = []
        for features in features_per_batch:
            box, image = self.density_estimator(features)
            box_scores.append(box)
            image_scores.append(image)

        box_locations = [batch_item["boxes"] for batch_item in regions]
        return box_locations, box_scores, image_scores
