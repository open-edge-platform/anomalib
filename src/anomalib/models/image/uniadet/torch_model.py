# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""UniADet zero-/few-shot detection, independently implemented from arXiv:2601.05552."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from anomalib.data import InferenceBatch
from anomalib.models.components import DynamicBufferMixin
from anomalib.models.components.filters import GaussianBlur2d

from .backbones import HierarchicalFeatures, UniADetBackbone
from .head import MSDCSHead


class UniADetModel(DynamicBufferMixin, nn.Module):
    """Frozen encoder, decoupled prototypes, and category-specific few-shot patch memory.

    The auxiliary head must be trained or loaded before prediction. Few-shot
    memory contains normal examples of one target category only and is stored in
    the state dict, so exported models do not need a reference-image directory.

    Args:
        backbone (str): ``"clip"``, ``"dinov2"`` or ``"dinov3"``.
        layers (tuple[int, ...]): One-based feature block numbers.
        pretrained (bool): Load pretrained encoder weights.
        image_size (int | None): Fixed square encoder resolution.
        temperature (float): Cosine-logit temperature.
        few_shot_weight (float): Legacy reference-distance weight; author protocol fixes its own weights.
        image_score_weight (float): Legacy pixel-peak weight; author protocol fixes its own weights.
        sigma (float): Gaussian smoothing standard deviation at input-image resolution.
        protocol (str): ``"legacy"`` preserves existing heads and scores. ``"author"``
            uses raw DINO features and published executable scoring, including a 256-square
            image-score pooling map. Requires a head trained for that protocol.

    Raises:
        ValueError: If fusion weights are outside ``[0, 1]`` or sigma is nonpositive.
    """

    def __init__(
        self,
        backbone: str = "dinov2",
        layers: tuple[int, ...] = (12, 15, 18, 21, 24),
        pretrained: bool = True,
        image_size: int | None = None,
        temperature: float = 0.07,
        few_shot_weight: float = 0.5,
        image_score_weight: float = 0.5,
        sigma: float = 4.0,
        protocol: str = "legacy",
    ) -> None:
        super().__init__()
        if protocol not in {"legacy", "author"}:
            msg = "protocol must be 'legacy' or 'author'."
            raise ValueError(msg)
        if protocol == "author" and (few_shot_weight != 0.5 or image_score_weight != 0.5):
            msg = "The author protocol fixes its fusion weights; custom weights require the legacy protocol."
            raise ValueError(msg)
        if not 0 <= few_shot_weight <= 1 or not 0 <= image_score_weight <= 1 or sigma <= 0:
            msg = "Fusion weights must be in [0, 1] and sigma must be positive."
            raise ValueError(msg)
        self.backbone_name = backbone
        self.layers = layers
        self.protocol = protocol
        self.backbone = UniADetBackbone(backbone, layers, pretrained, image_size, protocol=protocol)
        self.head = MSDCSHead(self.backbone.feature_dim, len(layers), temperature)
        self.few_shot_weight = few_shot_weight
        self.image_score_weight = image_score_weight
        self.blur = GaussianBlur2d(sigma=sigma)
        self.memory_bank: torch.Tensor
        self.head_trained: torch.Tensor
        self.register_buffer("memory_bank", torch.empty(len(layers), 0, self.backbone.feature_dim))
        self.register_buffer("head_trained", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def set_reference_images(self, images: torch.Tensor) -> None:
        """Replace the target category's memory with normal reference-image features.

        Args:
            images (torch.Tensor): Preprocessed normal images ``(K, 3, H, W)``.

        Raises:
            ValueError: If no reference images are provided.
        """
        if images.ndim != 4 or images.shape[0] == 0:
            msg = "At least one reference image is required in a (K, 3, H, W) tensor."
            raise ValueError(msg)
        features = self.backbone(images)
        # Lightning can initialize references in inference mode, but checkpoint restore writes into this buffer.
        with torch.inference_mode(mode=False):
            self.memory_bank = torch.stack([
                functional.normalize(patches.flatten(0, 1), dim=-1) for patches in features.patch_tokens
            ])

    @torch.inference_mode(mode=False)
    def clear_reference_images(self) -> None:
        """Remove category-specific memory before zero-shot inference or switching categories."""
        self.memory_bank = self.memory_bank.new_empty(len(self.layers), 0, self.backbone.feature_dim)

    def load_head(self, path: str | Path) -> None:
        """Load a trusted tensor-only auxiliary head produced by the reproduction script.

        Args:
            path (str | Path): Head checkpoint with configuration and tensor state.

        Raises:
            ValueError: If the checkpoint configuration differs from this model.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        expected = {
            "backbone": self.backbone_name,
            "layers": list(self.layers),
            "image_size": self.backbone.image_size,
            "temperature": self.head.temperature,
        }
        if self.protocol != "legacy":
            expected["protocol"] = self.protocol
        if checkpoint.get("config") != expected:
            msg = f"Head checkpoint configuration does not match {expected}."
            raise ValueError(msg)
        self.head.load_state_dict(checkpoint["state_dict"], strict=True)
        self.head_trained.fill_(value=True)

    def save_head(self, path: str | Path) -> None:
        """Save the trained head without bundling or redistributing backbone weights.

        Args:
            path (str | Path): Destination checkpoint path.
        """
        config = {
            "backbone": self.backbone_name,
            "layers": list(self.layers),
            "image_size": self.backbone.image_size,
            "temperature": self.head.temperature,
        }
        if self.protocol != "legacy":
            config["protocol"] = self.protocol
        torch.save(
            {
                "config": config,
                "state_dict": {key: value.detach().cpu() for key, value in self.head.state_dict().items()},
            },
            path,
        )

    def _reference_distances(self, features: HierarchicalFeatures) -> torch.Tensor:
        distances = []
        for layer, patches in enumerate(features.patch_tokens):
            similarities = functional.normalize(patches, dim=-1) @ self.memory_bank[layer].T
            distances.append(1 - similarities.amax(dim=-1))
        return (
            torch.stack(distances)
            .mean(dim=0)
            .reshape(
                -1,
                1,
                self.backbone.grid_size,
                self.backbone.grid_size,
            )
        )

    def forward(self, images: torch.Tensor) -> InferenceBatch:
        """Predict anomaly scores and maps using paper equations 8-12.

        Args:
            images (torch.Tensor): Normalized images ``(B, 3, H, W)``.

        Returns:
            InferenceBatch: Image scores ``(B,)`` and anomaly maps ``(B, 1, H, W)``.

        Raises:
            RuntimeError: If the auxiliary head has not been trained or loaded.
        """
        features = self.backbone(images)
        return self.predict_from_features(features)

    def predict_from_features(self, features: HierarchicalFeatures) -> InferenceBatch:
        """Score extracted features, allowing benchmark shot settings to share encoder computation.

        Args:
            features (HierarchicalFeatures): Tokens from this model's frozen encoder.

        Returns:
            InferenceBatch: Image scores ``(B,)`` and input-resolution maps ``(B, 1, H, W)``.

        Raises:
            RuntimeError: If the auxiliary head has not been trained or loaded.
        """
        if not torch.compiler.is_compiling() and not self.head_trained:
            msg = "UniADet requires a trained auxiliary head. Load weights or run tools/uniadet/train.py first."
            raise RuntimeError(msg)
        logits = self.head(features, self.backbone.grid_size)
        image_scores = torch.stack([image.softmax(dim=-1)[:, 1] for image in logits.image]).mean(dim=0)
        anomaly_map = torch.stack([pixel.softmax(dim=1)[:, 1:2] for pixel in logits.pixel]).mean(dim=0)
        if self.protocol == "author":
            return self._author_predictions(features, image_scores, anomaly_map)
        if self.memory_bank.shape[1]:
            anomaly_map = (1 - self.few_shot_weight) * anomaly_map + self.few_shot_weight * self._reference_distances(
                features,
            )
        anomaly_map = self.blur(
            functional.interpolate(
                anomaly_map,
                size=(self.backbone.image_size, self.backbone.image_size),
                mode="bilinear",
                align_corners=True,
            ),
        )
        pred_score = (1 - self.image_score_weight) * image_scores + self.image_score_weight * anomaly_map.flatten(
            1,
        ).amax(dim=1)
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    def _author_predictions(
        self,
        features: HierarchicalFeatures,
        image_scores: torch.Tensor,
        patch_scores: torch.Tensor,
    ) -> InferenceBatch:
        size = (self.backbone.image_size,) * 2
        anomaly_map = functional.interpolate(patch_scores, size=size, mode="bilinear", align_corners=False)
        anomaly_map = torch.sigmoid(15 * (anomaly_map - 0.4))
        if self.memory_bank.shape[1]:
            reference = functional.interpolate(
                self._reference_distances(features),
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            reference = torch.sigmoid(15 * (self.blur(reference) - 0.3))
            anomaly_map = 0.2 * anomaly_map + 0.8 * reference
        anomaly_map = self.blur(anomaly_map)
        score_map = functional.interpolate(anomaly_map, size=(256, 256), mode="bilinear", align_corners=False)
        peak_weight = 0.5 if self.memory_bank.shape[1] else 0.1
        score = (1 - peak_weight) * image_scores + peak_weight * score_map.flatten(1).amax(1)
        return InferenceBatch(pred_score=score, anomaly_map=anomaly_map)
