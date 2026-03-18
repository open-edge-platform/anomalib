# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model implementation for L2BT."""

from __future__ import annotations

import torch
from torch import nn

from anomalib.data import InferenceBatch

from .anomaly_map import L2BTAnomalyMapGenerator
from .students import FeatureProjectionMLP
from .teacher import FeatureExtractor


class L2BTModel(nn.Module):
    """PyTorch implementation of L2BT (teacher + two students)."""

    def __init__(
        self,
        layers: tuple[int, int] = (7, 11),
        blur_w_l: int = 5,
        blur_w_u: int = 7,
        blur_pad_l: int = 2,
        blur_pad_u: int = 3,
        blur_repeats_l: int = 5,
        blur_repeats_u: int = 3,
        topk_ratio: float = 0.001,
    ) -> None:
        """Initialize the L2BT model.

        Args:
            layers: Teacher transformer layers used for feature extraction.
            blur_w_l: Lower blur kernel width.
            blur_w_u: Upper blur kernel width.
            blur_pad_l: Lower blur padding.
            blur_pad_u: Upper blur padding.
            blur_repeats_l: Number of repetitions for the lower blur kernel.
            blur_repeats_u: Number of repetitions for the upper blur kernel.
            topk_ratio: Fraction of highest anomaly-map values used for image scoring.
        """
        super().__init__()

        self.layers = list(layers)
        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

        # Teacher is frozen and must remain in eval mode.
        self.teacher = FeatureExtractor(layers=self.layers).eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Students are trainable.
        self.backward_net = FeatureProjectionMLP(
            in_features=self.teacher.embed_dim,
            out_features=self.teacher.embed_dim,
        )
        self.forward_net = FeatureProjectionMLP(
            in_features=self.teacher.embed_dim,
            out_features=self.teacher.embed_dim,
        )

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
    def extract_teacher_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract frozen teacher features for the two selected ViT layers."""
        if images.ndim != 4:
            msg = f"Expected images with shape (B,C,H,W), got {tuple(images.shape)}"
            raise ValueError(msg)

        # Keep teacher in eval mode even when parent module is in train mode.
        self.teacher.eval()

        middle_patch, last_patch = self.teacher(images)
        return middle_patch, last_patch

    def predict_student_features(
        self,
        middle_patch: torch.Tensor,
        last_patch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the cross-layer mappings learned by the two student MLPs."""
        predicted_middle_patch = self.backward_net(last_patch)
        predicted_last_patch = self.forward_net(middle_patch)
        return predicted_middle_patch, predicted_last_patch

    def compute_losses(
        self,
        middle_patch: torch.Tensor,
        last_patch: torch.Tensor,
        predicted_middle_patch: torch.Tensor,
        predicted_last_patch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total loss plus the two directional losses used in the original code."""
        loss_middle = 1 - self.cos_sim(predicted_middle_patch, middle_patch).mean()
        loss_last = 1 - self.cos_sim(predicted_last_patch, last_patch).mean()
        loss = loss_middle + loss_last
        return loss, loss_middle, loss_last

    def _training_forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass used during training."""
        middle_patch, last_patch = self.extract_teacher_features(images)
        predicted_middle_patch, predicted_last_patch = self.predict_student_features(middle_patch, last_patch)
        loss, loss_middle, loss_last = self.compute_losses(
            middle_patch=middle_patch,
            last_patch=last_patch,
            predicted_middle_patch=predicted_middle_patch,
            predicted_last_patch=predicted_last_patch,
        )
        return {
            "loss": loss,
            "loss_middle": loss_middle,
            "loss_last": loss_last,
            "middle_patch": middle_patch,
            "last_patch": last_patch,
            "predicted_middle_patch": predicted_middle_patch,
            "predicted_last_patch": predicted_last_patch,
        }

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor] | InferenceBatch:
        """Run training or inference depending on module mode."""
        if self.training:
            return self._training_forward(images)

        middle_patch, last_patch = self.extract_teacher_features(images)
        output_size = images.shape[-2:]

        predicted_middle_patch, predicted_last_patch = self.predict_student_features(middle_patch, last_patch)

        anomaly_map, pred_score = self.anomaly_map_generator(
            middle_patch=middle_patch,
            last_patch=last_patch,
            predicted_middle_patch=predicted_middle_patch,
            predicted_last_patch=predicted_last_patch,
            output_size=output_size,
        )

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)