"""
SaveToCSV Callback
"""
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from pytorch_lightning import Callback, Trainer

from anomalib.models.base import ClassificationModule, SegmentationModule


class SaveToCSVCallback(Callback):
    """
    Callback that saves the inference results of a model. The callback generates a csv file that saves different
    performance metrics and results.
    """

    def __init__(self):
        """SaveToCSV callback"""

    def on_test_epoch_end(self, _trainer: Trainer, pl_module: Union[ClassificationModule, SegmentationModule]) -> None:
        """Save Results at the end of training
        Args:
            _trainer (Trainer): Pytorch lightning trainer object (unused)
            pl_module (LightningModule): Lightning modules derived from BaseAnomalyLightning object.
        """

        results = pl_module.results
        data_frame = pd.DataFrame(
            {
                "name": results.filenames,
                "true_label": results.true_labels,
                "pred_label": results.pred_labels.astype(int),
                "wrong_prediction": np.logical_xor(results.true_labels, results.pred_labels).astype(int),
            }
        )
        data_frame.to_csv(Path(pl_module.hparams.project.path) / "results.csv")
