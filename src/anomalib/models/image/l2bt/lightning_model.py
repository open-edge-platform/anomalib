# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning wrapper for L2BT."""

from __future__ import annotations

from itertools import chain
from typing import Any

import torch
from torchvision.transforms.v2 import Resize

from anomalib.models import AnomalibModule
from anomalib.pre_processing import PreProcessor

from .torch_model import L2BTModel


class L2BT(AnomalibModule):
    """AnomalibModule wrapper for L2BT."""

    def __init__(
        self,
        lr: float = 1e-4,
        image_size: int = 1036,
        load_pretrained: bool | None = None,
        **model_kwargs: Any,
    ) -> None:
        pre_processor = PreProcessor(transform=Resize((image_size, image_size)))
        super().__init__(pre_processor=pre_processor)

        self.save_hyperparameters(ignore=["pre_processor"])
        self.lr = lr
        self.model = L2BTModel(load_pretrained=load_pretrained, **model_kwargs)

    @property
    def learning_type(self) -> str:
        return "one_class"

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        return {}

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

    def training_step(self, batch: Any, batch_idx: int, *args: Any, **kwargs: Any) -> torch.Tensor:
        images = self._get_images(batch)
        out = self.model.training_forward(images)

        loss = out["loss"]
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=images.shape[0])
        self.log(
            "train_loss_middle",
            out["loss_middle"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=images.shape[0],
        )
        self.log(
            "train_loss_last",
            out["loss_last"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=images.shape[0],
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params=chain(self.model.backward_net.parameters(), self.model.forward_net.parameters()),
            lr=self.lr,
        )

    @staticmethod
    def _update_batch(batch: Any, pred_score: torch.Tensor, anomaly_map: torch.Tensor) -> Any:
        """Attach predictions to the original batch so anomalib post-processing can modify them."""
        if hasattr(batch, "update") and callable(batch.update):
            return batch.update(pred_score=pred_score, anomaly_map=anomaly_map)

        if isinstance(batch, dict):
            batch["pred_score"] = pred_score
            batch["anomaly_map"] = anomaly_map
            return batch

        # Fallback for mutable objects/dataclasses with writable attributes
        try:
            setattr(batch, "pred_score", pred_score)
            setattr(batch, "anomaly_map", anomaly_map)
            return batch
        except AttributeError as exc:
            raise TypeError(
                f"Unsupported batch type for inference output update: {type(batch)}. "
                "Expected a batch with .update(...), a dict, or writable attributes."
            ) from exc

    def _forward_inference(self, batch: Any) -> Any:
        images = self._get_images(batch)
        out = self.model(images)
        return self._update_batch(batch=batch, pred_score=out.pred_score, anomaly_map=out.anomaly_map)

    def validation_step(self, batch: Any, batch_idx: int, *args: Any, **kwargs: Any) -> Any:
        return self._forward_inference(batch)

    def test_step(self, batch: Any, batch_idx: int, *args: Any, **kwargs: Any) -> Any:
        return self._forward_inference(batch)

    def predict_step(self, batch: Any, batch_idx: int, *args: Any, **kwargs: Any) -> Any:
        return self._forward_inference(batch)