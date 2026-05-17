# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning wrapper for CFM (Cross-modal Feature Mapping)."""

from __future__ import annotations

from typing import Any

import torch
from torchvision.transforms.v2 import Resize

from anomalib import LearningType
from anomalib.models import AnomalibModule
from anomalib.pre_processing import PreProcessor

from .torch_model import CFMModel


class CFM(AnomalibModule):
    """AnomalibModule wrapper for CFM model."""

    def __init__(
        self,
        lr: float = 1e-4,
        rgb_backbone: str = "vit_base_patch8_224.dino",
        group_size: int = 128,
        num_group: int = 1024,
    ) -> None:
        """Initialize lightning module for CFM.

        Args:
            lr: Learning rate.
            rgb_backbone: Name of the backbone DINO for RGB.
            group_size: Dimension of the groups for PointTransformer.
            num_group: Number of groups for PointTransformer.
        """
        super().__init__()

        self.save_hyperparameters()
        self.lr = lr

        # Inizialization of core model
        self.model = CFMModel(
            rgb_backbone=rgb_backbone,
            group_size=group_size,
            num_group=num_group,
        )
        
    def configure_pre_processor(self, image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure the pre-processor dynamically based on config/data."""
        size = image_size if image_size is not None else (224, 224)
        return PreProcessor(transform=Resize(size))

    @property
    def learning_type(self) -> LearningType:
        """Returns the model's learning type (One-Class)."""
        return LearningType.ONE_CLASS

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Returns specific arguments for the trainer."""
        return {}

    @staticmethod
    def _get_data(batch: object) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract both the RGB image and the Point Cloud from multimodal batch."""
        if isinstance(batch, dict):
            rgb = batch.get("image")
            xyz = batch.get("point_cloud")
            if rgb is not None and xyz is not None:
                return rgb, xyz

        msg = "Tensor 'image' and 'point_cloud' not found in the batch."
        raise KeyError(msg)

    def training_step(self, batch: object, _batch_idx: int, *_args: object, **_kwargs: object) -> torch.Tensor:
        """Executes a training step evaluating the loss between multimodal projections."""
        rgb, xyz = self._get_data(batch)
        out = self.model(rgb, xyz)

        loss = out["loss"]
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=rgb.shape[0])
        return loss

    def validation_step(self, batch: object, _batch_idx: int, *_args: object, **_kwargs: object) -> object:
        """Executes a validation step and  uno step di validazione and attaches the anomaly maps to the batch."""
        rgb, xyz = self._get_data(batch)
        out = self.model(rgb, xyz)

        # Update the batch with inference results for the evaluation of Anomalib metrics
        return self._update_batch(batch, out.pred_score, out.anomaly_map)

    def test_step(self, batch: object, batch_idx: int, *_args: object, **_kwargs: object) -> object:
        """Executes a test step."""
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configuration of the optimizer (Adam) for trainable modules."""
        # Optimization of mapper parameters only (projection nets)
        return torch.optim.Adam(
            params=self.model.mapper_parameters(),
            lr=self.lr,
        )

    @staticmethod
    def _update_batch(batch: object, pred_score: torch.Tensor, anomaly_map: torch.Tensor) -> object:
        """Utility method to attach predictions to the original batch."""
        if isinstance(batch, dict):
            batch["pred_score"] = pred_score
            batch["anomaly_map"] = anomaly_map
        elif hasattr(batch, "update"):
            batch.update(pred_score=pred_score, anomaly_map=anomaly_map)
        return batch
