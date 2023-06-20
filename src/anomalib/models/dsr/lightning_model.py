"""DSR – A Dual Subspace Re-Projection Network for Surface Anomaly Detection

Paper https://link.springer.com/chapter/10.1007/978-3-031-19821-2_31
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from os.path import isfile

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from anomalib.models.components import AnomalyModule
from anomalib.models.dsr.anomaly_generator import DsrAnomalyGenerator
from anomalib.models.dsr.download_weights import DsrWeightDownloader
from anomalib.models.dsr.loss import DsrLoss
from anomalib.models.dsr.torch_model import DsrModel

__all__ = ["Dsr", "DsrLightning"]

logger = logging.getLogger(__name__)


class Dsr(AnomalyModule):
    """DSR: A Dual Subspace Re-Projection Network for Surface Anomaly Detection

    Args:
        anomaly_source_path (str | None): Path to folder that contains the anomaly source images. Random noise will
            be used if left empty.
    """

    def __init__(self, ckpt: str, anom_par: float = 0.2) -> None:
        super().__init__()

        # while "model < objective or end epoch" on train
        # else train upsampling module till epoch end

        self.anomaly_generator = DsrAnomalyGenerator()
        self.model = DsrModel(ckpt, anom_par)
        self.loss = DsrLoss()
        self.downloader = DsrWeightDownloader()
        self.anom_par: float = anom_par

        if not isfile("src/anomalib/models/dsr/vq_model_pretrained_128_4096.pckl"):
            logger.info("Pretrained weights not found.")
            self.downloader.download()
        else:
            logger.info("Pretrained checkpoint file found.")

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Training Step of DSR.

        Feeds the original image and the simulated anomaly mask during first phase. During
        second phase, feeds a generated anomalous image to train the upsampling module.

        Args:
            batch (dict[str, str | Tensor]): Batch containing image filename, image, label and mask

        Returns:
            Loss dictionary
        """
        del args, kwargs  # These variables are not used.

        input_image = batch["image"]
        # Create anomaly masks
        anomaly_mask = self.anomaly_generator.augment_batch(input_image)
        # Generate model prediction
        recon_nq_hi, recon_nq_lo, qu_hi, qu_lo, gen_img, seg, anomaly_mask = self.model(input_image, anomaly_mask)
        # Compute loss
        loss = self.loss(recon_nq_hi, recon_nq_lo, qu_hi, qu_lo, input_image, gen_img, seg, anomaly_mask)

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Validation step of DSR. The Softmax predictions of the anomalous class are used as anomaly map.

        Args:
            batch (dict[str, str | Tensor]): Batch of input images

        Returns:
            Dictionary to which predicted anomaly maps have been added.
        """
        del args, kwargs  # These variables are not used.

        prediction, anomaly_scores = self.model(batch["image"])
        batch["anomaly_maps"] = prediction
        batch["pred_scores"] = anomaly_scores
        return batch


class DsrLightning(Dsr):
    """DSR: A Dual Subspace Re-Projection Network for Surface Anomaly Detection

    Args:
        hparams (DictConfig | ListConfig): Model parameters
    """

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(ckpt=hparams.model.ckpt_path, anom_par=hparams.model.anom_par)
        self.hparams: DictConfig | ListConfig  # type: ignore
        self.save_hyperparameters(hparams)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the Adam optimizer. Do not train the discrete model! (or the upsmapler for the time being)"""
        return torch.optim.Adam(params=self.model.parameters(), lr=self.hparams.model.lr)
