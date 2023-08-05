"""Class used to compute metrics for ensemble predictions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Tuple

from omegaconf import DictConfig, ListConfig

from anomalib.data import TaskType
from anomalib.models.ensemble.post_processing.postprocess import EnsemblePostProcess
from anomalib.utils.metrics import AnomalibMetricCollection, create_metric_collection

logger = logging.getLogger(__name__)


class EnsembleMetrics(EnsemblePostProcess):
    """
    Class that works as block in ensemble pipeline used to calcualte metrics.

    Args:
        config: Configurable parameters object.
        image_threshold: Threshold used for image metrics.
        pixel_threshold: Threshold used for pixel metrics.
    """

    def __init__(self, config: DictConfig | ListConfig, image_threshold: float, pixel_threshold: float) -> None:
        super().__init__(final_compute=True, name="metrics")

        self.image_metrics, self.pixel_metrics = self.configure_ensemble_metrics(
            config.dataset.task,
            config.metrics.get("image", None),
            config.metrics.get("pixel", None),
        )

        # set threshold for metrics that require it
        self.image_metrics.set_threshold(image_threshold)
        self.pixel_metrics.set_threshold(pixel_threshold)

        self.image_metrics.cpu()
        self.pixel_metrics.cpu()

    @staticmethod
    def configure_ensemble_metrics(
        task: TaskType = TaskType.SEGMENTATION,
        image_metrics: List[str] | None = None,
        pixel_metrics: List[str] | None = None,
    ) -> Tuple[AnomalibMetricCollection, AnomalibMetricCollection]:
        """
        Configure image and pixel metrics and put them into a collection.

        Args:
            task: Task type of the current run.
            image_metrics: List of image-level metric names.
            pixel_metrics: List of pixel-level metric names.

        Returns:
            Image-metrics collection and pixel-metrics collection
        """
        image_metric_names = [] if image_metrics is None else image_metrics

        pixel_metric_names: list[str]
        if pixel_metrics is None:
            pixel_metric_names = []
        elif task == TaskType.CLASSIFICATION:
            pixel_metric_names = []
            logger.warning(
                "Cannot perform pixel-level evaluation when task type is classification. "
                "Ignoring the following pixel-level metrics: %s",
                pixel_metrics,
            )
        else:
            pixel_metric_names = pixel_metrics

        image_metrics = create_metric_collection(image_metric_names, "image_")
        pixel_metrics = create_metric_collection(pixel_metric_names, "pixel_")

        return image_metrics, pixel_metrics

    def process(self, data: dict) -> dict:
        """
        Compute metrics specified in config for given ensemble results.

        Args:
            data: Joined batch of results produced by model ensemble.

        Returns:
            Unchanged input.
        """
        self.image_metrics.update(data["pred_scores"], data["label"].int())
        if "mask" in data.keys() and "anomaly_maps" in data.keys():
            self.pixel_metrics.update(data["anomaly_maps"], data["mask"].int())

        return data

    def compute(self) -> dict:
        """
        Compute metrics for entire ensemble.

        Returns:
            Dictionary containing calculated metric data.
        """
        out = {}
        for name, metric in self.image_metrics.items():
            out[name] = metric.compute().item()

        if self.pixel_metrics.update_called:
            for name, metric in self.pixel_metrics.items():
                out[name] = metric.compute().item()

        return out


def log_metrics(metric_dict: dict[str, float]) -> None:
    """
    Log computed metrics.

    Args:
        metric_dict: Dictionary containing all metrics info.
    """
    for name, value in metric_dict.items():
        logger.info("%s: %f", name, value)
