"""Anomalib pre-processing module."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from lightning import Callback, LightningModule, Trainer
from torch import nn
from torchvision.transforms.v2 import Transform

from anomalib.data.dataclasses.torch.base import Batch
from anomalib.deploy.utils import get_exportable_transform


class PreProcessor(nn.Module, Callback):
    """Anomalib pre-processor.

    This class serves as both a PyTorch module and a Lightning callback, handling
    the application of transforms to data batches during different stages of
    training, validation, testing, and prediction.

    Args:
        train_transform (Transform | None): Transform to apply during training.
        val_transform (Transform | None): Transform to apply during validation.
        test_transform (Transform | None): Transform to apply during testing.
        transform (Transform | None): General transform to apply if stage-specific
            transforms are not provided.

    Raises:
        ValueError: If both `transform` and any of the stage-specific transforms
            are provided simultaneously.

    Notes:
        If only `transform` is provided, it will be used for all stages (train, val, test).

    Examples:
        >>> from torchvision.transforms.v2 import Compose, Resize, ToTensor
        >>> from anomalib.pre_processing import PreProcessor

        >>> # Define transforms
        >>> train_transform = Compose([Resize((224, 224)), ToTensor()])
        >>> val_transform = Compose([Resize((256, 256)), CenterCrop((224, 224)), ToTensor()])

        >>> # Create PreProcessor with stage-specific transforms
        >>> pre_processor = PreProcessor(
        ...     train_transform=train_transform,
        ...     val_transform=val_transform
        ... )

        >>> # Create PreProcessor with a single transform for all stages
        >>> common_transform = Compose([Resize((224, 224)), ToTensor()])
        >>> pre_processor_common = PreProcessor(transform=common_transform)

        >>> # Use in a Lightning module
        >>> class MyModel(LightningModule):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.pre_processor = PreProcessor(...)
        ...
        ...     def configure_callbacks(self):
        ...         return [self.pre_processor]
        ...
        ...     def training_step(self, batch, batch_idx):
        ...         # The pre_processor will automatically apply the correct transform
        ...         processed_batch = self.pre_processor(batch)
        ...         # Rest of the training step
    """

    def __init__(
        self,
        train_transform: Transform | None = None,
        val_transform: Transform | None = None,
        test_transform: Transform | None = None,
        transform: Transform | None = None,
    ) -> None:
        super().__init__()

        if transform and any([train_transform, val_transform, test_transform]):
            msg = (
                "`transforms` cannot be used together with `train_transform`, `val_transform`, `test_transform`.\n"
                "If you want to apply the same transform to the training, validation and test data, "
                "use only `transforms`. \n"
                "Otherwise, specify transforms for training, validation and test individually."
            )
            raise ValueError(msg)

        self.train_transform = train_transform or transform
        self.val_transform = get_exportable_transform(val_transform or transform)
        self.test_transform = get_exportable_transform(test_transform or transform)

        self.current_transform = self.train_transform  # Default to train transform

    def forward(self, batch: Batch | torch.Tensor) -> Batch | torch.Tensor:
        """Apply transforms to the batch."""
        if self.current_transform:
            if isinstance(batch, Batch):
                image, gt_mask = self.current_transform(batch.image, batch.gt_mask)
                batch.update(image=image, gt_mask=gt_mask)
            else:
                batch = self.current_transform(batch)
        return batch

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Set the current transform to the train transform."""
        del trainer, pl_module  # Unused parameters
        self.current_transform = self.train_transform

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
    ) -> None:
        """Apply transforms to the training batch."""
        del trainer, pl_module, batch_idx  # Unused parameters
        self(batch)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Set the current transform to the validation transform."""
        del trainer, pl_module  # Unused parameters
        self.current_transform = self.val_transform

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Apply transforms to the validation batch."""
        del trainer, pl_module, batch_idx, dataloader_idx  # Unused parameters
        self(batch)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Set the current transform to the test transform."""
        del trainer, pl_module  # Unused parameters
        self.current_transform = self.test_transform

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Apply transforms to the test batch."""
        del trainer, pl_module, batch_idx, dataloader_idx  # Unused parameters
        self(batch)

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Set the current transform to the test transform."""
        del trainer, pl_module  # Unused parameters
        self.current_transform = self.test_transform

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Apply transforms to the predict batch."""
        del trainer, pl_module, batch_idx, dataloader_idx  # Unused parameters
        self(batch)
