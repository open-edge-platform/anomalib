# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Helper functions to generate plots of various metrics.

This module provides utility functions for generating ROC-style plots and other
visualization helpers used by metrics in Anomalib.
"""

import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from anomalib.utils.deprecation import deprecate


def plot_score_histogram(
    scores: torch.Tensor,
    labels: torch.Tensor,
    bins: int = 30,
    good_color: str = "skyblue",
    bad_color: str = "salmon",
    xlabel: str = "Score",
    ylabel: str = "Relative Count",
    title: str = "Score Histogram",
    legend_labels: tuple[str, str] = ("Good", "Bad"),
) -> tuple[Figure, Axes]:
    """Plot a histogram of scores using two colors for good and bad images.

    Args:
        scores (torch.Tensor): 1D tensor of scores for all images.
        labels (torch.Tensor): 1D tensor of binary labels (0=good, 1=bad).
        bins (int, optional): Number of histogram bins. Defaults to ``30``.
        good_color (str, optional): Color for good images. Defaults to "skyblue".
        bad_color (str, optional): Color for bad images. Defaults to "salmon".
        xlabel (str, optional): Label for x-axis. Defaults to "Score".
        ylabel (str, optional): Label for y-axis. Defaults to "Relative Count".
        title (str, optional): Title of the plot. Defaults to "Score Histogram".
        legend_labels (tuple[str, str], optional): Labels for legend. Defaults to ("Good", "Bad").

    Returns:
        tuple[Figure, Axes]: Tuple containing the figure and its main axis.
    """
    fig, axis = plt.subplots()
    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    good_scores = scores[labels == 0]
    bad_scores = scores[labels == 1]

    axis.hist(
        good_scores,
        bins=bins,
        color=good_color,
        alpha=0.7,
        label=legend_labels[0],
        density=True,
    )
    axis.hist(
        bad_scores,
        bins=bins,
        color=bad_color,
        alpha=0.7,
        label=legend_labels[1],
        density=True,
    )

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    return fig, axis


def plot_metric_curve(
    x_vals: torch.Tensor,
    y_vals: torch.Tensor,
    auc: torch.Tensor,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    loc: str,
    title: str,
    metric_name: str = "AUC",
    sample_points: int = 1000,
) -> tuple[Figure, Axes]:
    """Generate a metric curve plot (e.g. ROC, PR, PRO) with x values plotted against y values.

    The function creates a matplotlib figure with a single axis showing the curve
    defined by ``x_vals`` and ``y_vals``. If the number of points exceeds
    ``sample_points``, the data is subsampled to improve plotting performance.

    Args:
        x_vals (torch.Tensor): Values to plot on x-axis.
        y_vals (torch.Tensor): Values to plot on y-axis.
        auc (torch.Tensor): Area under curve value to display in legend (e.g. AUROC, AUPR, AUPRO).
        xlim (tuple[float, float]): Display range for x-axis as ``(min, max)``.
        ylim (tuple[float, float]): Display range for y-axis as ``(min, max)``.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        loc (str): Legend location. See matplotlib documentation for valid values:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
        title (str): Title of the plot.
        metric_name (str, optional): Name of the metric to display in legend. Defaults to "AUC".
        sample_points (int, optional): Maximum number of points to plot. Data will
            be subsampled if it exceeds this value. Defaults to ``1000``.

    Returns:
        tuple[Figure, Axes]: Tuple containing the figure and its main axis.

    Example:
        >>> import torch
        >>> x = torch.linspace(0, 1, 100)
        >>> y = x ** 2
        >>> auc = torch.tensor(0.5)
        >>> fig, ax = plot_metric_curve(
        ...     x_vals=x,
        ...     y_vals=y,
        ...     auc=auc,
        ...     xlim=(0, 1),
        ...     ylim=(0, 1),
        ...     xlabel="False Positive Rate",
        ...     ylabel="True Positive Rate",
        ...     loc="lower right",
        ...     title="ROC Curve",
        ...     metric_name="AUROC",
        ... )
    """
    fig, axis = plt.subplots()

    x_vals = x_vals.detach().cpu()
    y_vals = y_vals.detach().cpu()

    if sample_points < x_vals.size(0):
        possible_idx = range(x_vals.size(0))
        interval = len(possible_idx) // sample_points

        idx = [0]  # make sure to start at first point
        idx.extend(possible_idx[::interval])
        idx.append(possible_idx[-1])  # also include last point

        idx = torch.tensor(
            idx,
            device=x_vals.device,
        )
        x_vals = torch.index_select(x_vals, 0, idx)
        y_vals = torch.index_select(y_vals, 0, idx)

    axis.plot(
        x_vals,
        y_vals,
        color="darkorange",
        figure=fig,
        lw=2,
        label=f"{metric_name}: {auc.detach().cpu():0.2f}",
    )

    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend(loc=loc)
    axis.set_title(title)
    return fig, axis


@deprecate(since="2.1.0", remove="2.3.0", use="plot_metric_curve")
def plot_figure(
    x_vals: torch.Tensor,
    y_vals: torch.Tensor,
    auc: torch.Tensor,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    loc: str,
    title: str,
    metric_name: str = "AUC",
    sample_points: int = 1000,
) -> tuple[Figure, Axes]:
    """Generate a ROC-style plot with x values plotted against y values.

    .. deprecated:: 2.1.0
        This function is deprecated and will be removed in v2.3.0.
        Use :func:`plot_metric_curve` instead.

    The function creates a matplotlib figure with a single axis showing the curve
    defined by ``x_vals`` and ``y_vals``. If the number of points exceeds
    ``sample_points``, the data is subsampled to improve plotting performance.

    Args:
        x_vals (torch.Tensor): Values to plot on x-axis.
        y_vals (torch.Tensor): Values to plot on y-axis.
        auc (torch.Tensor): Area under curve value to display in legend.
        xlim (tuple[float, float]): Display range for x-axis as ``(min, max)``.
        ylim (tuple[float, float]): Display range for y-axis as ``(min, max)``.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        loc (str): Legend location. See matplotlib documentation for valid values:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
        title (str): Title of the plot.
        metric_name (str, optional): Name of the metric to display in legend. Defaults to "AUC".
        sample_points (int, optional): Maximum number of points to plot. Data will
            be subsampled if it exceeds this value. Defaults to ``1000``.

    Returns:
        tuple[Figure, Axes]: Tuple containing the figure and its main axis.

    Example:
        >>> import torch
        >>> x = torch.linspace(0, 1, 100)
        >>> y = x ** 2
        >>> auc = torch.tensor(0.5)
        >>> fig, ax = plot_figure(
        ...     x_vals=x,
        ...     y_vals=y,
        ...     auc=auc,
        ...     xlim=(0, 1),
        ...     ylim=(0, 1),
        ...     xlabel="False Positive Rate",
        ...     ylabel="True Positive Rate",
        ...     loc="lower right",
        ...     title="ROC Curve",
        ... )
    """
    return plot_metric_curve(
        x_vals=x_vals,
        y_vals=y_vals,
        auc=auc,
        xlim=xlim,
        ylim=ylim,
        xlabel=xlabel,
        ylabel=ylabel,
        loc=loc,
        title=title,
        metric_name=metric_name,
        sample_points=sample_points,
    )
