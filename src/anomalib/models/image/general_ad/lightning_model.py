# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning implementation of GeneralAD."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch import optim
from torchvision.transforms.v2 import Compose, InterpolationMode, Normalize, Resize

from anomalib import LearningType
from anomalib.metrics import AUROC, Evaluator, F1Score
from anomalib.models.components import AnomalibModule
from anomalib.pre_processing import PreProcessor

from .torch_model import FakeFeatureType, GeneralADModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lightning.pytorch.utilities.types import STEP_OUTPUT

    from anomalib.data import Batch
    from anomalib.post_processing import PostProcessor
    from anomalib.visualization import Visualizer

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
        backbone: str = "vit_large_patch14_dinov2.lvd142m",
        layers: Sequence[int] = (24,),
        hidden_dim: int = 2048,
        lr: float = 5e-4,
        lr_decay_factor: float = 0.2,
        weight_decay: float = 1e-5,
        epochs: int = 160,
        noise_std: float = 0.25,
        dsc_layers: int = 1,
        dsc_heads: int = 4,
        dsc_dropout: float = 0.1,
        num_fake_patches: int = -1,
        fake_feature_type: FakeFeatureType = "random",
        top_k: int = 10,
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

        image_size = (self.input_size[0], self.input_size[1]) if self.input_size is not None else (518, 518)

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
            image_size = (518, 518)
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

    @staticmethod
    def configure_evaluator() -> Evaluator:
        """Configure validation and test metrics for checkpoint selection.

        Upstream GeneralAD selects the best checkpoint using image-level AUROC on
        the validation loop. In our MVTec reproduction, validation is configured
        as ``SAME_AS_TEST`` to mirror the upstream behavior, so monitoring
        ``image_AUROC`` here reproduces the same selection rule.
        """
        val_image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        test_image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
        pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False)
        pixel_f1score = F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False)
        val_metrics = [val_image_auroc]
        test_metrics = [test_image_auroc, image_f1score, pixel_auroc, pixel_f1score]
        return Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)

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
