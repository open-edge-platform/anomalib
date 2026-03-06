# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model implementation for L2BT."""

from __future__ import annotations

import torch
import torch.nn as nn

from anomalib.data import InferenceBatch

from .teacher import FeatureExtractor
from .students import FeatureProjectionMLP

from .anomaly_map import L2BTAnomalyMapGenerator


class L2BTModel(nn.Module):
    """PyTorch implementation of L2BT (teacher + two students)."""

    def __init__(
        self,
        checkpoint_folder: str = "./checkpoints/checkpoints_visa",
        class_name: str = "candle",
        label: str = "final_model",
        epochs_no: int = 50,
        batch_size: int = 4,
        layers: tuple[int, int] = (7, 11),
        blur_w_l: int = 5,
        blur_w_u: int = 7,
        blur_pad_l: int = 2,
        blur_pad_u: int = 3,
        blur_repeats_l: int = 5,
        blur_repeats_u: int = 3,
        topk_ratio: float = 0.001,
    ) -> None:
        super().__init__()

        self.layers = list(layers)

        # Teacher (frozen)
        self.teacher = FeatureExtractor(layers=self.layers).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Students
        self.backward_net = FeatureProjectionMLP(
            in_features=self.teacher.embed_dim,
            out_features=self.teacher.embed_dim,
        ).eval()
        self.forward_net = FeatureProjectionMLP(
            in_features=self.teacher.embed_dim,
            out_features=self.teacher.embed_dim,
        ).eval()

        for p in self.backward_net.parameters():
            p.requires_grad = False
        for p in self.forward_net.parameters():
            p.requires_grad = False

        # Load checkpoints (same naming as the original script)
        forward_net_path = (
            f"{checkpoint_folder}/{class_name}/"
            f"forward_net_{label}_{class_name}_{epochs_no}ep_{batch_size}bs.pth"
        )
        backward_net_path = (
            f"{checkpoint_folder}/{class_name}/"
            f"backward_net_{label}_{class_name}_{epochs_no}ep_{batch_size}bs.pth"
        )

        self.forward_net.load_state_dict(torch.load(forward_net_path, weights_only=False))
        self.backward_net.load_state_dict(torch.load(backward_net_path, weights_only=False))

        # Anomaly map generator (encapsulates blur + topk score)
        self.anomaly_map_generator = L2BTAnomalyMapGenerator(
            patch_size=int(self.teacher.patch_size),
            blur_w_l=blur_w_l,
            blur_w_u=blur_w_u,
            blur_pad_l=blur_pad_l,
            blur_pad_u=blur_pad_u,
            blur_repeats_l=blur_repeats_l,
            blur_repeats_u=blur_repeats_u,
            topk_ratio=topk_ratio,
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> InferenceBatch:
        """Return Anomalib InferenceBatch(pred_score, anomaly_map)."""
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape (B,C,H,W), got {tuple(images.shape)}")

        output_size = images.shape[-2:]

        middle_patch, last_patch = self.teacher(images)
        predicted_middle_patch = self.backward_net(last_patch)
        predicted_last_patch = self.forward_net(middle_patch)

        anomaly_map, pred_score = self.anomaly_map_generator(
            middle_patch=middle_patch,
            last_patch=last_patch,
            predicted_middle_patch=predicted_middle_patch,
            predicted_last_patch=predicted_last_patch,
            output_size=output_size,
        )

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)