# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning wrapper for L2BT."""

from __future__ import annotations

from itertools import chain

import torch
from torchvision.transforms.v2 import Resize

from anomalib import LearningType
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
        checkpoint_folder: str = "./checkpoints/checkpoints_visa",
        class_name: str = "candle",
        label: str = "final_model",
        epochs_no: int = 50,
        batch_size: int = 4,
        layers: tuple[int, int] = (7, 11),
        blur_w_l: int = 5,
        blur_w_u: int = 7,
        blur_pad_l: int = 2,
        blur_pad_u: int = 3,
        blur_repeats_l: int = 5,
        blur_repeats_u: int = 3,
        topk_ratio: float = 0.001,
        strict_checkpoint_load: bool = True,
    ) -> None:
        """Initialize the L2BT lightning module.

        Args:
            lr: Learning rate for student optimization.
            image_size: Input image size used by the pre-processor.
            load_pretrained: Whether to load pretrained student checkpoints.
            checkpoint_folder: Directory containing pretrained student checkpoints.
            class_name: Dataset category name.
            label: Label identifying the checkpoint files.
            epochs_no: Number of training epochs used for the checkpoints.
            batch_size: Batch size used during training.
            layers: Teacher transformer layers used for feature extraction.
            blur_w_l: Lower blur kernel width.
            blur_w_u: Upper blur kernel width.
            blur_pad_l: Lower blur padding.
            blur_pad_u: Upper blur padding.
            blur_repeats_l: Number of repetitions for the lower blur kernel.
            blur_repeats_u: Number of repetitions for the upper blur kernel.
            topk_ratio: Fraction of highest anomaly-map values used for image scoring.
            strict_checkpoint_load: Whether checkpoint loading should be strict.
        """
        pre_processor = PreProcessor(transform=Resize((image_size, image_size)))
        super().__init__(pre_processor=pre_processor)

        self.save_hyperparameters(ignore=["pre_processor"])
        self.lr = lr
        self.model = L2BTModel(
            checkpoint_folder=checkpoint_folder,
            class_name=class_name,
            label=label,
            epochs_no=epochs_no,
            batch_size=batch_size,
            layers=layers,
            blur_w_l=blur_w_l,
            blur_w_u=blur_w_u,
            blur_pad_l=blur_pad_l,
            blur_pad_u=blur_pad_u,
            blur_repeats_l=blur_repeats_l,
            blur_repeats_u=blur_repeats_u,
            topk_ratio=topk_ratio,
            load_pretrained=load_pretrained,
            strict_checkpoint_load=strict_checkpoint_load,
        )

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model."""
        return LearningType.ONE_CLASS

    @property
    def trainer_arguments(self) -> dict[str, int | float | str | bool]:
        """Return trainer arguments for the model."""
        return {}

    @staticmethod
    def _get_images(batch: object) -> torch.Tensor:
        """Extract the image tensor from a batch."""
        if hasattr(batch, "image"):
            return batch.image  # type: ignore[attr-defined]
        if isinstance(batch, dict):
            if "image" in batch:
                return batch["image"]
            if "img" in batch:
                return batch["img"]
        msg = "Could not find image tensor in batch (expected .image or ['image'] or ['img'])."
        raise KeyError(msg)

    def training_step(self, batch: object, _batch_idx: int, *_args: object, **_kwargs: object) -> torch.Tensor:
        """Compute the training loss for a batch."""
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
        """Configure the optimizer used during training."""
        return torch.optim.Adam(
            params=chain(self.model.backward_net.parameters(), self.model.forward_net.parameters()),
            lr=self.lr,
        )

    @staticmethod
    def _update_batch(batch: object, pred_score: torch.Tensor, anomaly_map: torch.Tensor) -> object:
        """Attach predictions to the original batch so anomalib post-processing can modify them."""
        if hasattr(batch, "update") and callable(batch.update):
            batch.update(pred_score=pred_score, anomaly_map=anomaly_map)  # type: ignore[attr-defined]
            return batch

        if isinstance(batch, dict):
            batch["pred_score"] = pred_score
            batch["anomaly_map"] = anomaly_map
            return batch

        try:
            batch.pred_score = pred_score  # type: ignore[attr-defined]
            batch.anomaly_map = anomaly_map  # type: ignore[attr-defined]
        except AttributeError as exc:
            msg = (
                f"Unsupported batch type for inference output update: {type(batch)}. "
                "Expected a batch with .update(...), a dict, or writable attributes."
            )
            raise TypeError(msg) from exc
        else:
            return batch

    def _forward_inference(self, batch: object) -> object:
        """Run inference and attach predictions to the batch."""
        images = self._get_images(batch)
        out = self.model(images)
        return self._update_batch(batch=batch, pred_score=out.pred_score, anomaly_map=out.anomaly_map)

    def validation_step(self, batch: object, _batch_idx: int, *_args: object, **_kwargs: object) -> object:
        """Run a validation step."""
        return self._forward_inference(batch)

    def test_step(self, batch: object, batch_idx: int, *_args: object, **_kwargs: object) -> object:
        """Run a test step."""
        del batch_idx
        return self._forward_inference(batch)

    def predict_step(self, batch: object, batch_idx: int, dataloader_idx: int = 0) -> object:
        """Run a prediction step."""
        del batch_idx, dataloader_idx
        return self._forward_inference(batch)
