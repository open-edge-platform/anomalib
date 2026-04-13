# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning wrapper for L2BT."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import chain
from typing import TYPE_CHECKING, Any

import torch
from torchvision.transforms.v2 import Compose, InterpolationMode, Normalize, Resize

from anomalib import LearningType
from anomalib.data.transforms import SquarePad
from anomalib.models.components import AnomalibModule
from anomalib.pre_processing import PreProcessor

from .torch_model import L2BTModel

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT

    from anomalib.data import Batch
    from anomalib.metrics import Evaluator
    from anomalib.post_processing import PostProcessor
    from anomalib.visualization import Visualizer


class L2BT(AnomalibModule):
    """AnomalibModule wrapper for L2BT."""

    def __init__(
        self,
        lr: float = 1e-4,
        layers: Sequence[int] = (7, 11),
        blur_w_l: int = 5,
        blur_w_u: int = 7,
        blur_pad_l: int = 2,
        blur_pad_u: int = 3,
        blur_repeats_l: int = 5,
        blur_repeats_u: int = 3,
        topk_ratio: float = 0.001,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        """Initialize the L2BT lightning module.

        Args:
            lr (float): Learning rate for student optimization.
            layers (Sequence[int]): Teacher transformer layers used for feature extraction.
                Accepts any sequence (list or tuple) of exactly two indices.
            blur_w_l (int): Lower blur kernel width.
            blur_w_u (int): Upper blur kernel width.
            blur_pad_l (int): Lower blur padding.
            blur_pad_u (int): Upper blur padding.
            blur_repeats_l (int): Number of repetitions for the lower blur kernel.
            blur_repeats_u (int): Number of repetitions for the upper blur kernel.
            topk_ratio (float): Fraction of highest anomaly-map values used for image scoring.
            pre_processor (PreProcessor | bool, optional): Pre-processor instance or
                flag to use default. Defaults to ``True``.
            post_processor (PostProcessor | bool, optional): Post-processor instance
                or flag to use default. Defaults to ``True``.
            evaluator (Evaluator | bool, optional): Evaluator instance or flag to
                use default. Defaults to ``True``.
            visualizer (Visualizer | bool, optional): Visualizer instance or flag to
                use default. Defaults to ``True``.
        """
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.lr = lr
        self.model = L2BTModel(
            layers=layers,
            blur_w_l=blur_w_l,
            blur_w_u=blur_w_u,
            blur_pad_l=blur_pad_l,
            blur_pad_u=blur_pad_u,
            blur_repeats_l=blur_repeats_l,
            blur_repeats_u=blur_repeats_u,
            topk_ratio=topk_ratio,
        )

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model."""
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_pre_processor(image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure the default pre-processor for L2BT.

        The original L2BT pipeline applies: SquarePad (edge replication) →
        Resize (bicubic) → ImageNet normalization.

        Args:
            image_size (tuple[int, int] | None, optional): Target image size.
                Defaults to ``(224, 224)``.

        Returns:
            PreProcessor: Configured pre-processor with the L2BT transform pipeline.
        """
        image_size = image_size or (224, 224)
        return PreProcessor(
            transform=Compose([
                SquarePad(),
                Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        )

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return trainer arguments for the model."""
        return {}

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Compute the training loss for a batch.

        Args:
            batch (Batch): Input batch containing images and metadata.
            args: Additional positional arguments (unused).
            kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Dictionary containing the training loss.
        """
        del args, kwargs  # These variables are not used.

        out = self.model(batch.image)

        loss = out["loss"]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_middle", out["loss_middle"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_loss_last", out["loss_last"], on_step=True, on_epoch=True, prog_bar=False)
        return {"loss": loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer used during training."""
        return torch.optim.Adam(
            params=chain(self.model.backward_net.parameters(), self.model.forward_net.parameters()),
            lr=self.lr,
        )

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Run a validation step.

        Args:
            batch (Batch): Input batch containing images and metadata.
            args: Additional positional arguments (unused).
            kwargs: Additional keyword arguments (unused).

        Returns:
            STEP_OUTPUT: Updated batch with predictions.
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())
