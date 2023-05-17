"""Implements custom trainer for Anomalib."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
import warnings

from pytorch_lightning import Trainer

from anomalib.data import TaskType
from anomalib.models.components.base.anomaly_module import AnomalyModule
from anomalib.post_processing import NormalizationMethod
from anomalib.post_processing.visualizer import VisualizationMode
from anomalib.trainer.connectors import (
    CheckpointConnector,
    MetricsConnector,
    PostProcessingConnector,
    ThresholdingConnector,
    VisualizationConnector,
    get_normalizer,
)
from anomalib.trainer.loops.one_class import FitLoop, PredictionLoop, TestLoop, ValidationLoop
from anomalib.utils.metrics import BaseAnomalyThreshold

log = logging.getLogger(__name__)
# warnings to ignore in trainer
warnings.filterwarnings(
    "ignore", message="torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead"
)


class AnomalibTrainer(Trainer):
    """Anomalib trainer.

    Note:
        Refer to PyTorch Lightning's Trainer for a list of parameters for details on other Trainer parameters.

    Args:
        image_threshold_method (dict): Thresholding method. If None, adaptive thresholding is used.
        pixel_threshold_method (dict): Thresholding method. If None, adaptive thresholding is used.
        normalization_method (NormalizationMethod): Normalization method
        image_metrics (list[str] | None): Image metrics to compute. Defaults to None.
        pixel_metrics (list[str] | None): Pixel metrics to compute. Defaults to None.
        visualization_mode (VisualizationMode): Visualization mode. Defaults to VisualizationMode.FULL.
        show_images (bool): Whether to show images during visualization. Defaults to False.
        log_images (bool): Whether to log images to available loggers during visualization. Defaults to False.
        task_type (TaskType): Task type. Defaults to TaskType.SEGMENTATION.
        **kwargs: Additional keyword arguments to pass to the Trainer.
    """

    def __init__(
        self,
        image_threshold: dict | None = None,  # TODO change from dict to BaseAnomalyThreshold in CLI
        pixel_threshold: dict | None = None,
        normalization_method: NormalizationMethod = NormalizationMethod.MIN_MAX,
        image_metrics: list[str] | None = None,
        pixel_metrics: list[str] | None = None,
        visualization_mode: VisualizationMode = VisualizationMode.FULL,
        show_images: bool = False,
        log_images: bool = False,
        task_type: TaskType = TaskType.SEGMENTATION,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._checkpoint_connector = CheckpointConnector(self, kwargs.get("resume_from_checkpoint", None))

        self.lightning_module: AnomalyModule  # for mypy

        self.fit_loop = FitLoop(min_epochs=kwargs.get("min_epochs", 0), max_epochs=kwargs.get("max_epochs", None))
        self.validate_loop = ValidationLoop()
        self.test_loop = TestLoop()
        self.predict_loop = PredictionLoop()

        self.task_type = task_type
        # these are part of the trainer as they are used in the metrics-manager, post-processor and thresholder
        self.image_threshold: BaseAnomalyThreshold | None = None
        self.pixel_threshold: BaseAnomalyThreshold | None = None

        self.thresholding_connector = ThresholdingConnector(
            trainer=self,
            image_threshold_method=image_threshold,
            pixel_threshold_method=pixel_threshold,
        )
        self.post_processing_connector = PostProcessingConnector(trainer=self)
        self.normalization_connector = get_normalizer(trainer=self, normalization_method=normalization_method)
        self.visualization_connector = VisualizationConnector(
            trainer=self,
            mode=visualization_mode,
            show_images=show_images,
            log_images=log_images,
        )
        self.metrics_connector = MetricsConnector(
            trainer=self, image_metrics=image_metrics, pixel_metrics=pixel_metrics
        )
