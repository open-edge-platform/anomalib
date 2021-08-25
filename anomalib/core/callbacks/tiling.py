"""Callback for tiling image"""
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

from anomalib.datasets.tiler import Tiler


class TilingCallback(Callback):
    """
    Callback that tiles and untiles images to help detect small defects. Tiles batches of images during training and
    validation, before the batch is presented as input to the model. Untiles images and model predictions after the
    forward pass is completed.
    """

    def __init__(self, hparams):
        self.tiler = Tiler(hparams.dataset.tiling.tile_size, hparams.dataset.tiling.stride)

    def on_train_batch_start(
        self,
        _trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
        batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the train batch begins."""
        batch["image"] = self.tiler.tile(batch["image"])

    def on_validation_batch_start(
        self,
        _trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
        batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the validation batch begins."""
        batch["image"] = self.tiler.tile(batch["image"])

    def on_validation_batch_end(
        self,
        _trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends."""
        if isinstance(outputs, Dict):
            outputs["anomaly_maps"] = self.tiler.untile(outputs["anomaly_maps"])
            outputs["images"] = self.tiler.untile(outputs["images"])

    def on_test_batch_start(
        self,
        _trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
        batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the test batch begins."""
        batch["image"] = self.tiler.tile(batch["image"])

    def on_test_batch_end(
        self,
        _trainer: pl.Trainer,
        _pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends."""
        if outputs is not None:
            outputs["anomaly_maps"] = self.tiler.untile(outputs["anomaly_maps"])
            outputs["images"] = self.tiler.untile(outputs["images"])
