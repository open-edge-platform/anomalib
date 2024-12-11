"""U-Flow: A U-shaped Normalizing Flow for Anomaly Detection with Unsupervised Threshold.

https://arxiv.org/pdf/2211.12353.pdf
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import torch
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import LRScheduler
from torchvision.transforms.v2 import Compose, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .loss import UFlowLoss
from .torch_model import UflowModel

logger = logging.getLogger(__name__)

__all__ = ["Uflow"]


class Uflow(AnomalibModule):
    """Uflow model.

    Args:
        backbone (str): Backbone name.
        flow_steps (int): Number of flow steps.
        affine_clamp (float): Affine clamp.
        affine_subnet_channels_ratio (float): Affine subnet channels ratio.
        permute_soft (bool): Whether to use soft permutation.
    """

    def __init__(
        self,
        backbone: str = "mcait",
        flow_steps: int = 4,
        affine_clamp: float = 2.0,
        affine_subnet_channels_ratio: float = 1.0,
        permute_soft: bool = False,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        """Uflow model.

        Args:
            backbone (str): Backbone name.
            flow_steps (int): Number of flow steps.
            affine_clamp (float): Affine clamp.
            affine_subnet_channels_ratio (float): Affine subnet channels ratio.
            permute_soft (bool): Whether to use soft permutation.
            pre_processor (PreProcessor, optional): Pre-processor for the model.
                This is used to pre-process the input data before it is passed to the model.
                Defaults to ``None``.
            post_processor (PostProcessor, optional): Post-processor for the model.
                This is used to post-process the output data after it is passed to the model.
                Defaults to ``None``.
            evaluator (Evaluator, optional): Evaluator for the model.
                This is used to evaluate the model.
                Defaults to ``True``.
            visualizer (Visualizer, optional): Visualizer for the model.
                This is used to visualize the model.
                Defaults to ``True``.
        """
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        if self.input_size is None:
            msg = "Input size is required for UFlow model."
            raise ValueError(msg)

        self.backbone = backbone
        self.flow_steps = flow_steps
        self.affine_clamp = affine_clamp
        self.affine_subnet_channels_ratio = affine_subnet_channels_ratio
        self.permute_soft = permute_soft

        self.model = UflowModel(
            input_size=self.input_size,
            backbone=self.backbone,
            flow_steps=self.flow_steps,
            affine_clamp=self.affine_clamp,
            affine_subnet_channels_ratio=self.affine_subnet_channels_ratio,
            permute_soft=self.permute_soft,
        )
        self.loss = UFlowLoss()

    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Default pre-processor for UFlow."""
        if image_size is not None:
            logger.warning("Image size is not used in UFlow. The input image size is determined by the model.")
        transform = Compose([
            Resize((448, 448), antialias=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return PreProcessor(transform=transform)

    def configure_optimizers(self) -> tuple[list[LightningOptimizer], list[LRScheduler]]:
        """Return optimizer and scheduler."""
        # Optimizer
        # values used in paper: bottle: 0.0001128999, cable: 0.0016160391, capsule: 0.0012118892, carpet: 0.0012118892,
        # grid: 0.0000362248, hazelnut: 0.0013268899, leather: 0.0006124724, metal_nut: 0.0008148858,
        # pill: 0.0010756100, screw: 0.0004155987, tile: 0.0060457548, toothbrush: 0.0001287313,
        # transistor: 0.0011212904, wood: 0.0002466546, zipper: 0.0000455247
        optimizer = torch.optim.Adam([{"params": self.parameters(), "initial_lr": 1e-3}], lr=1e-3, weight_decay=1e-5)

        # Scheduler for slowly reducing learning rate
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.4,
            total_iters=25000,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:  # noqa: ARG002 | unused arguments
        """Training step."""
        z, ljd = self.model(batch.image)
        loss = self.loss(z, ljd)
        self.log_dict({"loss": loss}, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:  # noqa: ARG002 | unused arguments
        """Validation step."""
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return EfficientAD trainer arguments."""
        return {"num_sanity_val_steps": 0}

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS
