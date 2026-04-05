# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning implementation of GeneralAD."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim
from torchvision.transforms.v2 import Compose, InterpolationMode, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import FakeFeatureType, GeneralADModel

__all__ = ["GeneralAD"]


class GeneralAD(AnomalibModule):
    """GeneralAD Lightning module.

    Args:
        backbone: Timm backbone used as frozen feature extractor.
        layers: Backbone stages or transformer blocks used for patch features.
        hidden_dim: Hidden size of the patch discriminator.
        lr: Optimizer learning rate.
        lr_decay_factor: Final cosine annealing ratio.
        weight_decay: Optimizer weight decay.
        epochs: Number of epochs used to parameterize the LR scheduler.
        noise_std: Standard deviation for synthetic anomalous features.
        dsc_layers: Number of attention blocks in the discriminator.
        dsc_heads: Number of attention heads in the discriminator.
        dsc_dropout: Dropout rate in the discriminator.
        pool_size: Patch pooling size for CNN backbones.
        image_size: Input image size after preprocessing.
        num_fake_patches: Maximum number of perturbed patches per image.
        fake_feature_type: Strategy for generating pseudo anomalies.
        top_k: Number of highest patch scores to average for image scoring.
        pre_trained: Whether to use pretrained backbone weights.
        pre_processor: Optional anomalib pre-processor.
        post_processor: Optional anomalib post-processor.
        evaluator: Optional anomalib evaluator.
        visualizer: Optional anomalib visualizer.
    """

    def __init__(
        self,
        backbone: str = "vit_tiny_patch16_224",
        layers: Sequence[int] = (9, 10, 11, 12),
        hidden_dim: int = 1024,
        lr: float = 1e-4,
        lr_decay_factor: float = 1e-2,
        weight_decay: float = 1e-5,
        epochs: int = 100,
        noise_std: float = 0.015,
        dsc_layers: int = 1,
        dsc_heads: int = 12,
        dsc_dropout: float = 0.0,
        pool_size: int = 3,
        image_size: tuple[int, int] | int = (256, 256),
        num_fake_patches: int = 64,
        fake_feature_type: FakeFeatureType = "copy_out_and_attn",
        top_k: int = -1,
        pre_trained: bool = True,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.lr = lr
        self.lr_decay_factor = lr_decay_factor
        self.weight_decay = weight_decay
        self.epochs = epochs

        self.model = GeneralADModel(
            backbone=backbone,
            layers=layers,
            hidden_dim=hidden_dim,
            noise_std=noise_std,
            dsc_layers=dsc_layers,
            dsc_heads=dsc_heads,
            dsc_dropout=dsc_dropout,
            pool_size=pool_size,
            image_size=image_size,
            num_fake_patches=num_fake_patches,
            fake_feature_type=fake_feature_type,
            top_k=top_k,
            pre_trained=pre_trained,
        )

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | int | None = None,
    ) -> PreProcessor:
        """Configure the default pre-processor."""
        if image_size is None:
            image_size = (256, 256)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        return PreProcessor(
            transform=Compose([
                Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        )

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Run the self-supervised GeneralAD training step."""
        del args, kwargs
        loss = self.model.compute_loss(batch.image)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Generate validation predictions."""
        del args, kwargs
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Default trainer arguments for GeneralAD."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and cosine LR schedule."""
        optimizer = optim.AdamW(self.model.discriminator.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=self.lr * self.lr_decay_factor,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    @property
    def learning_type(self) -> LearningType:
        """GeneralAD is a one-class anomaly detection model."""
        return LearningType.ONE_CLASS
