"""Anomaly Score Normalization Callback that uses min-max normalization."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Any, Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT


class MinMaxNormalizationCallback(Callback):
    """Callback that normalizes the image-level and pixel-level anomaly scores using min-max normalization."""

    def on_test_start(self, _trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when the test begins."""
        pl_module.image_metrics.F1.threshold = 0.5
        pl_module.pixel_metrics.F1.threshold = 0.5

    def on_validation_batch_end(
        self,
        _trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the validation batch ends, update the min and max observed values."""
        if "anomaly_maps" in outputs.keys():
            pl_module.min_max(outputs["anomaly_maps"])
        else:
            pl_module.min_max(outputs["pred_scores"])

    def on_test_batch_end(
        self,
        _trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends, normalizes the predicted scores and anomaly maps."""
        self._normalize_batch(outputs, pl_module)

    def on_predict_batch_end(
        self,
        _trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict,
        _batch: Any,
        _batch_idx: int,
        _dataloader_idx: int,
    ) -> None:
        """Called when the predict batch ends, normalizes the predicted scores and anomaly maps."""
        self._normalize_batch(outputs, pl_module)

    def _normalize_batch(self, outputs, pl_module):
        """Normalize a batch of predictions."""
        stats = pl_module.min_max
        outputs["pred_scores"] = self._normalize(
            outputs["pred_scores"], pl_module.image_threshold.value, stats.min, stats.max
        )
        if "anomaly_maps" in outputs.keys():
            outputs["anomaly_maps"] = self._normalize(
                outputs["anomaly_maps"], pl_module.pixel_threshold.value, stats.min, stats.max
            )

    def _normalize(self, predictions, threshold, min_val, max_val):
        """Normalize the predictions using mean normalization centered at the threshold."""
        normalized_predictions = ((predictions - threshold) / (max_val - min_val)) + 0.5
        normalized_predictions = torch.minimum(normalized_predictions, torch.tensor(1))  # pylint: disable=not-callable
        normalized_predictions = torch.maximum(normalized_predictions, torch.tensor(0))  # pylint: disable=not-callable
        return normalized_predictions
