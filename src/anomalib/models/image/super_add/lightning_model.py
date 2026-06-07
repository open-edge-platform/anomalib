# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" """

import logging
from collections.abc import Sequence
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

from anomalib import LearningType, PrecisionType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import SuperADDModel

logger = logging.getLogger(__name__)


class SuperADD(MemoryBankMixin, AnomalibModule):
    """ """

    def __init__(
        self,
        weights_path: str = None,
        backbone: str = "dinov3_vits16",  # "dinov3_vit7b16",
        layers: Sequence[str] = [3, 5, 7, 10],  # [7, 15, 23, 31],
        patch_size: int = 448,
        patch_overlap: int = 16,
        precision: str | PrecisionType = PrecisionType.FLOAT32,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        if weights_path is None:
            msg = f"DINOv3 ({backbone})Weights path must be provided for SuperADD."
            raise ValueError(msg)

        self.model: SuperADDModel = SuperADDModel(
            backbone=backbone,
            layers=layers,
            weights_path=weights_path,
            patch_batch_size=patch_size,
            patch_overlap=patch_overlap,
        )

        if isinstance(precision, str):
            precision = PrecisionType(precision.lower())

        if precision == PrecisionType.FLOAT16:
            self.model = self.model.bfloat16()
        elif precision == PrecisionType.FLOAT32:
            self.model = self.model.float()
        else:
            msg = f"""Unsupported precision type: {precision}.
            Supported types are: {PrecisionType.FLOAT16}, {PrecisionType.FLOAT32}."""
            raise ValueError(msg)

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
        center_crop_size: tuple[int, int] | None = None,
    ) -> PreProcessor:
        """Configure the default pre-processor for PatchCore.

        If valid center_crop_size is provided, the pre-processor will
        also perform center cropping, according to the paper.

        Args:
            image_size (tuple[int, int] | None, optional): Target size for
                resizing. Defaults to ``(256, 256)``.
            center_crop_size (tuple[int, int] | None, optional): Size for center
                cropping. Defaults to ``None``.

        Returns:
            PreProcessor: Configured pre-processor instance.

        Raises:
            ValueError: If at least one dimension of ``center_crop_size`` is larger
                than correspondent ``image_size`` dimension.

        Example:
            >>> pre_processor = Patchcore.configure_pre_processor(
            ...     image_size=(256, 256)
            ... )
            >>> transformed_image = pre_processor(image)
        """
        image_size = image_size or (448, 448)

        if center_crop_size is not None:
            if center_crop_size[0] > image_size[0] or center_crop_size[1] > image_size[1]:
                msg = f"Center crop size {center_crop_size} cannot be larger than image size {image_size}."
                raise ValueError(msg)
            transform = Compose([
                Resize(image_size, antialias=True),
                CenterCrop(center_crop_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = Compose([
                Resize(image_size, antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        return PreProcessor(transform=transform)

    @staticmethod
    def configure_optimizers() -> None:
        """Configure optimizers.

        Returns:
            None: PatchCore requires no optimization.
        """
        return

    def training_step(self, batch: Batch, *args, **kwargs) -> None:
        """Generate feature embedding of the batch.

        Args:
            batch (Batch): Input batch containing image and metadata
            *args: Additional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            torch.Tensor: Dummy loss tensor for Lightning compatibility

        Note:
            The method stores embeddings in ``self.embeddings`` for later use in
            ``fit()``.
        """
        del args, kwargs  # These variables are not used.
        _ = self.model(batch.image)
        # Return a dummy loss tensor
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        """Apply subsampling to the embedding collected from the training set.

        This method:
        1. Applies coreset subsampling to reduce memory requirements
        """
        logger.info("Applying core-set subsampling to get the embedding.")
        self.model.subsample_embedding()

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Generate predictions for a batch of images.

        Args:
            batch (Batch): Input batch containing images and metadata
            *args: Additional arguments (unused)
            **kwargs: Additional keyword arguments (unused)

        Returns:
            STEP_OUTPUT: Batch with added predictions

        Note:
            Predictions include anomaly maps and scores computed using nearest
            neighbor search.
        """
        # These variables are not used.
        del args, kwargs

        # Get anomaly maps and predicted scores from the model.
        predictions = self.model(batch.image)

        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Get default trainer arguments for PatchCore.

        Returns:
            dict[str, Any]: Trainer arguments
                - ``gradient_clip_val``: ``0`` (no gradient clipping needed)
                - ``max_epochs``: ``1`` (single pass through training data)
                - ``num_sanity_val_steps``: ``0`` (skip validation sanity checks)
                - ``devices``: ``1`` (only single gpu supported)
        """
        return {"gradient_clip_val": 0, "max_epochs": 1, "num_sanity_val_steps": 0, "devices": 1}

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type.

        Returns:
            LearningType: Always ``LearningType.ONE_CLASS`` as PatchCore only
                trains on normal samples
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        """Configure the default post-processor.

        Returns:
            PostProcessor: Post-processor for one-class models that
                converts raw scores to anomaly predictions
        """
        return PostProcessor()
