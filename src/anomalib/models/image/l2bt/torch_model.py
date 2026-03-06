# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model implementation for L2BT."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from anomalib.data import InferenceBatch

from .anomaly_map import L2BTAnomalyMapGenerator
from .students import FeatureProjectionMLP
from .teacher import FeatureExtractor


class L2BTModel(nn.Module):
    """PyTorch implementation of L2BT (teacher + two students).

    This class now supports both:
    - training from scratch inside anomalib
    - inference with optional loading of pre-trained student checkpoints
    """

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
        load_pretrained: bool | None = None,
        strict_checkpoint_load: bool = True,
    ) -> None:
        super().__init__()

        self.layers = list(layers)
        self.checkpoint_folder = checkpoint_folder
        self.class_name = class_name
        self.label = label
        self.epochs_no = epochs_no
        self.batch_size = batch_size
        self.load_pretrained = load_pretrained
        self.strict_checkpoint_load = strict_checkpoint_load
        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

        # Teacher is always frozen, exactly as in the original training code.
        self.teacher = FeatureExtractor(layers=self.layers).eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Students are trainable by default. This is required for anomalib training.
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

        self._maybe_load_students()

    def _checkpoint_paths(self) -> tuple[Path, Path]:
        checkpoint_dir = Path(self.checkpoint_folder) / self.class_name
        forward_path = checkpoint_dir / (
            f"forward_net_{self.label}_{self.class_name}_{self.epochs_no}ep_{self.batch_size}bs.pth"
        )
        backward_path = checkpoint_dir / (
            f"backward_net_{self.label}_{self.class_name}_{self.epochs_no}ep_{self.batch_size}bs.pth"
        )
        return forward_path, backward_path

    def _maybe_load_students(self) -> None:
        forward_path, backward_path = self._checkpoint_paths()
        checkpoints_exist = forward_path.exists() and backward_path.exists()

        should_load = self.load_pretrained
        if should_load is None:
            should_load = checkpoints_exist

        if not should_load:
            return

        if not checkpoints_exist:
            raise FileNotFoundError(
                "Requested loading pre-trained L2BT checkpoints, but at least one file was not found. "
                f"Expected: {forward_path} and {backward_path}"
            )

        self.forward_net.load_state_dict(
            torch.load(forward_path, map_location="cpu", weights_only=False),
            strict=self.strict_checkpoint_load,
        )
        self.backward_net.load_state_dict(
            torch.load(backward_path, map_location="cpu", weights_only=False),
            strict=self.strict_checkpoint_load,
        )

    @torch.no_grad()
    def extract_teacher_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract frozen teacher features for the two selected ViT layers."""
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape (B,C,H,W), got {tuple(images.shape)}")
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

    def training_forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass used during training inside anomalib."""
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

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> InferenceBatch:
        """Return anomalib InferenceBatch(pred_score, anomaly_map)."""
        output_size = images.shape[-2:]

        middle_patch, last_patch = self.extract_teacher_features(images)
        predicted_middle_patch, predicted_last_patch = self.predict_student_features(middle_patch, last_patch)

        anomaly_map, pred_score = self.anomaly_map_generator(
            middle_patch=middle_patch,
            last_patch=last_patch,
            predicted_middle_patch=predicted_middle_patch,
            predicted_last_patch=predicted_last_patch,
            output_size=output_size,
        )

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
