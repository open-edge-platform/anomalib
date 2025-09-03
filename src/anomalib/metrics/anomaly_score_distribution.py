# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compute statistics of anomaly score distributions.

This module provides the ``AnomalyScoreDistribution`` class, which computes the mean
and standard deviation statistics of anomaly scores.
Statistics are computed for both image-level and pixel-level scores.
The ``plot`` method generates a histogram of anomaly scores,
separated by label, to visualize score distributions for normal and abnormal samples.

The class tracks:
    - Image-level statistics: Mean and std of image anomaly scores
    - Pixel-level statistics: Mean and std of pixel anomaly maps

Example:
    >>> from anomalib.metrics import AnomalyScoreDistribution
    >>> import torch
    >>> # Create sample data
    >>> scores = torch.tensor([0.1, 0.2, 0.15])  # Image anomaly scores
    >>> maps = torch.tensor([[0.1, 0.2], [0.15, 0.25]])  # Pixel anomaly maps
    >>> labels = torch.tensor([0, 1, 0])  # Binary labels
    >>> # Initialize and compute stats
    >>> dist = AnomalyScoreDistribution()
    >>> dist.update(anomaly_scores=scores, anomaly_maps=maps, labels=labels)
    >>> image_mean, image_std, pixel_mean, pixel_std = dist.compute()
    >>> fig, title = dist.plot()

Note:
    The input scores and maps are log-transformed before computing statistics.
    Image-level scores, pixel-level maps, and labels are optional inputs.
"""

import torch
from matplotlib.figure import Figure
from torchmetrics import Metric

from .utils import plot_score_histogram


class AnomalyScoreDistribution(Metric):
    """Compute distribution statistics of anomaly scores.

    This class tracks and computes the mean and standard deviation of anomaly
    scores. Statistics are computed for both image-level scores and pixel-level
    anomaly maps.

    The metric maintains internal state to accumulate scores, anomaly maps,
    and labels across batches before computing final statistics.

    Example:
        >>> dist = AnomalyScoreDistribution()
        >>> # Update with batch of scores
        >>> scores = torch.tensor([0.1, 0.2, 0.3])
        >>> dist.update(anomaly_scores=scores)
        >>> # Compute statistics
        >>> img_mean, img_std, pix_mean, pix_std = dist.compute()
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the metric states.

        Args:
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.anomaly_maps: list[torch.Tensor] = []
        self.anomaly_scores: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []

        self.add_state("image_mean", torch.empty(0), persistent=True)
        self.add_state("image_std", torch.empty(0), persistent=True)
        self.add_state("pixel_mean", torch.empty(0), persistent=True)
        self.add_state("pixel_std", torch.empty(0), persistent=True)

        self.image_mean = torch.empty(0)
        self.image_std = torch.empty(0)
        self.pixel_mean = torch.empty(0)
        self.pixel_std = torch.empty(0)

    def update(
        self,
        *args,
        anomaly_scores: torch.Tensor | None = None,
        anomaly_maps: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """Update the internal state with new scores and maps.

        Args:
            *args: Unused positional arguments.
            anomaly_scores: Batch of image-level anomaly scores.
            anomaly_maps: Batch of pixel-level anomaly maps.
            labels: Batch of binary labels.
            **kwargs: Unused keyword arguments.
        """
        del args, kwargs  # These variables are not used.

        if anomaly_maps is not None:
            self.anomaly_maps.append(anomaly_maps)
        if anomaly_scores is not None:
            self.anomaly_scores.append(anomaly_scores)
        if labels is not None:
            self.labels.append(labels)

    def compute(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute distribution statistics from accumulated scores and maps.

        Returns:
            tuple containing:
                - image_mean: Mean of log-transformed image anomaly scores
                - image_std: Standard deviation of log-transformed image scores
                - pixel_mean: Mean of log-transformed pixel anomaly maps
                - pixel_std: Standard deviation of log-transformed pixel maps
        """
        anomaly_scores = torch.hstack(self.anomaly_scores)
        anomaly_scores = torch.log(anomaly_scores)

        self.image_mean = anomaly_scores.mean()
        self.image_std = anomaly_scores.std()

        if self.anomaly_maps:
            anomaly_maps = torch.vstack(self.anomaly_maps)
            anomaly_maps = torch.log(anomaly_maps).cpu()

            self.pixel_mean = anomaly_maps.mean(dim=0).squeeze()
            self.pixel_std = anomaly_maps.std(dim=0).squeeze()

        return self.image_mean, self.image_std, self.pixel_mean, self.pixel_std

    def plot(
        self,
        bins: int = 30,
        good_color: str = "skyblue",
        bad_color: str = "salmon",
        xlabel: str = "Score",
        ylabel: str = "Relative Count",
        title: str = "Score Histogram",
        legend_labels: tuple[str, str] = ("Good", "Bad"),
    ) -> tuple[Figure, str]:
        """Generate a histogram of scores.

        Args:
            bins (int, optional): Number of histogram bins. Defaults to 30.
            good_color (str, optional): Color for good samples. Defaults to "skyblue".
            bad_color (str, optional): Color for bad samples. Defaults to "salmon".
            xlabel (str, optional): Label for the x-axis. Defaults to "Score".
            ylabel (str, optional): Label for the y-axis. Defaults to "Relative Count".
            title (str, optional): Title of the plot. Defaults to "Score Histogram".
            legend_labels (tuple[str, str], optional): Legend labels for good and bad samples.
                Defaults to ("Good", "Bad").

        Returns:
            tuple[Figure, str]: Tuple containing both the figure and the figure
                title to be used for logging

        Raises:
            ValueError: If no anomaly scores or labels are available.
        """
        if len(self.anomaly_scores) == 0:
            msg = "No anomaly scores available."
            raise ValueError(msg)
        if len(self.labels) == 0:
            msg = "No labels available."
            raise ValueError(msg)

        fig, _ = plot_score_histogram(
            scores=torch.hstack(self.anomaly_scores),
            labels=torch.hstack(self.labels),
            bins=bins,
            good_color=good_color,
            bad_color=bad_color,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            legend_labels=legend_labels,
        )

        return fig, title
