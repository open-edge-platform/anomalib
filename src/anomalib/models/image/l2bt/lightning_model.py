# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning wrapper for L2BT (inference-only integration)."""

from __future__ import annotations

from typing import Any

import torch
from torchvision.transforms.v2 import Resize

from anomalib.data import InferenceBatch
from anomalib.models import AnomalibModule
from anomalib.pre_processing import PreProcessor

from .torch_model import L2BTModel


class L2BT(AnomalibModule):
    """AnomalibModule wrapper for L2BT.

    This integration supports inference only.
    Training is intentionally not implemented.
    """

    def __init__(self, **model_kwargs: Any) -> None:
        """Initialize the L2BT module."""
        pre_processor = PreProcessor(transform=Resize((224, 224)))
        super().__init__(pre_processor=pre_processor)

        self.save_hyperparameters(ignore=["pre_processor"])
        self.model = L2BTModel(**model_kwargs)

    @property
    def learning_type(self) -> str:
        return "one_class"

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        return {}

    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("Training is not implemented for L2BT in this integration.")

    @staticmethod
    def _get_images(batch: Any) -> torch.Tensor:
        """Support both dataclass batches (batch.image) and dict batches."""
        if hasattr(batch, "image"):
            return batch.image
        if isinstance(batch, dict):
            if "image" in batch:
                return batch["image"]
            if "img" in batch:
                return batch["img"]
        raise KeyError("Could not find image tensor in batch (expected .image or ['image'] or ['img']).")

    def _forward_inference(self, batch: Any) -> InferenceBatch:
        """Run model forward and return only standard InferenceBatch fields."""
        images = self._get_images(batch)
        out = self.model(images)

        return InferenceBatch(
            pred_score=out.pred_score,
            anomaly_map=out.anomaly_map,
        )

    def validation_step(self, batch: Any, batch_idx: int, *args: Any, **kwargs: Any) -> InferenceBatch:
        return self._forward_inference(batch)

    def test_step(self, batch: Any, batch_idx: int, *args: Any, **kwargs: Any) -> InferenceBatch:
        return self._forward_inference(batch)

    def predict_step(self, batch: Any, batch_idx: int, *args: Any, **kwargs: Any) -> InferenceBatch:
        return self._forward_inference(batch)