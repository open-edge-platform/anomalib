"""FastFlow Lightning Model Implementation.

https://arxiv.org/abs/2111.07677
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import AUROC, Evaluator, F1Score
from anomalib.models.components import AnomalyModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor

from .loss import FastflowLoss
from .torch_model import FastflowModel


class Fastflow(AnomalyModule):
    """PL Lightning Module for the FastFlow algorithm.

    Args:
        backbone (str): Backbone CNN network
            Defaults to ``resnet18``.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.
        flow_steps (int, optional): Flow steps.
            Defaults to ``8``.
        conv3x3_only (bool, optinoal): Use only conv3x3 in fast_flow model.
            Defaults to ``False``.
        hidden_ratio (float, optional): Ratio to calculate hidden var channels.
            Defaults to ``1.0``.
        pre_processor (PreProcessor, optional): Pre-processor for the model.
            This is used to pre-process the input data before it is passed to the model.
            Defaults to ``None``.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pre_trained: bool = True,
        flow_steps: int = 8,
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | None = None,
        evaluator: Evaluator | bool = True,
    ) -> None:
        super().__init__(pre_processor=pre_processor, post_processor=post_processor, evaluator=evaluator)
        if self.input_size is None:
            msg = "Fastflow needs input size to build torch model."
            raise ValueError(msg)

        self.backbone = backbone
        self.pre_trained = pre_trained
        self.flow_steps = flow_steps
        self.conv3x3_only = conv3x3_only
        self.hidden_ratio = hidden_ratio

        self.model = FastflowModel(
            input_size=self.input_size,
            backbone=self.backbone,
            pre_trained=self.pre_trained,
            flow_steps=self.flow_steps,
            conv3x3_only=self.conv3x3_only,
            hidden_ratio=self.hidden_ratio,
        )
        self.loss = FastflowLoss()

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the training step input and return the loss.

        Args:
            batch (batch: dict[str, str | torch.Tensor]): Input batch
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT: Dictionary containing the loss value.
        """
        del args, kwargs  # These variables are not used.

        hidden_variables, jacobians = self.model(batch.image)
        loss = self.loss(hidden_variables, jacobians)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step and return the anomaly map.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT | None: batch dictionary containing anomaly-maps.
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return FastFlow trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers for each decoder.

        Returns:
            Optimizer: Adam optimizer for each decoder
        """
        return optim.Adam(
            params=self.model.parameters(),
            lr=0.001,
            weight_decay=0.00001,
        )

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type of the model.
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_evaluator() -> Evaluator:
        """Default evaluator.

        Override in subclass for model-specific evaluator behaviour.
        """
        # val metrics (needed for early stopping)
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
        val_metrics = [image_auroc, pixel_auroc]

        # test_metrics
        image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
        image_f1score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
        pixel_auroc = AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_")
        pixel_f1score = F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_")
        test_metrics = [image_auroc, image_f1score, pixel_auroc, pixel_f1score]
        return Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)
