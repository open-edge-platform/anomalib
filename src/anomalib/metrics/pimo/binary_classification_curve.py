# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Binary classification curve.

This module provides functionality to compute binary classification matrices at
multiple thresholds. The thresholds are shared across all instances/images, but
binary classification metrics are computed independently for each instance/image.

The binary classification matrix contains:
- True Positives (TP)
- False Positives (FP)
- False Negatives (FN)
- True Negatives (TN)

Example:
    >>> import torch
    >>> from anomalib.metrics.pimo.binary_classification_curve import (
    ...     binary_classification_curve
    ... )
    >>> scores = torch.rand(10, 100)  # 10 images, 100 pixels each
    >>> gts = torch.randint(0, 2, (10, 100)).bool()  # Binary ground truth
    >>> thresholds = torch.linspace(0, 1, 10)  # 10 thresholds
    >>> curves = binary_classification_curve(scores, gts, thresholds)
    >>> curves.shape
    torch.Size([10, 10, 2, 2])
"""

import logging
from enum import Enum

import torch

from . import _validate

logger = logging.getLogger(__name__)


class ThresholdMethod(Enum):
    """Methods for selecting threshold sequences.

    Available methods:
        - ``GIVEN``: Use provided thresholds
        - ``MINMAX_LINSPACE``: Linear spacing between min and max scores
        - ``MEAN_FPR_OPTIMIZED``: Optimize based on mean false positive rate
    """

    GIVEN = "given"
    MINMAX_LINSPACE = "minmax-linspace"
    MEAN_FPR_OPTIMIZED = "mean-fpr-optimized"


def binary_classification_curve(
    scores_batch: torch.Tensor,
    gts_batch: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    """Compute binary classification matrices for a batch of images.

    Note:
        Predicted positives are determined by ``score >= thresh``

    Args:
        scores_batch: Anomaly scores of shape ``(N, D)``
        gts_batch: Binary ground truth of shape ``(N, D)``
        thresholds: Sequence of thresholds in ascending order ``(K,)``

    Returns:
        Binary classification matrix curves of shape ``(N, K, 2, 2)``
        where:

        - ``[..., 1, 1]``: True Positives (TP)
        - ``[..., 0, 1]``: False Positives (FP)
        - ``[..., 1, 0]``: False Negatives (FN)
        - ``[..., 0, 0]``: True Negatives (TN)

        The counts are per-instance (e.g. number of pixels in each image).
        Thresholds are shared across instances.

    Example:
        >>> scores = torch.rand(10, 100)  # 10 images, 100 pixels each
        >>> gts = torch.randint(0, 2, (10, 100)).bool()
        >>> thresholds = torch.linspace(0, 1, 10)
        >>> curves = binary_classification_curve(scores, gts, thresholds)
        >>> curves.shape
        torch.Size([10, 10, 2, 2])
    """
    _validate.is_scores_batch(scores_batch)
    _validate.is_gts_batch(gts_batch)
    _validate.is_same_shape(scores_batch, gts_batch)
    _validate.is_valid_threshold(thresholds)

    n, d = scores_batch.shape
    k = thresholds.shape[0]

    neg_inf = torch.tensor(-torch.inf, device=scores_batch.device, dtype=scores_batch.dtype)

    gts_batch = gts_batch.to(device=scores_batch.device)
    pos_scores = torch.where(gts_batch, scores_batch, neg_inf)
    neg_scores = torch.where(~gts_batch, scores_batch, neg_inf)

    pos_scores_sorted, _ = torch.sort(pos_scores, dim=1)
    neg_scores_sorted, _ = torch.sort(neg_scores, dim=1)

    thresholds = thresholds.to(device=scores_batch.device, dtype=scores_batch.dtype)
    thresh_exp = thresholds.unsqueeze(1).expand(k, n).contiguous().T.contiguous()

    pos_idx = torch.searchsorted(pos_scores_sorted, thresh_exp, side="left")
    neg_idx = torch.searchsorted(neg_scores_sorted, thresh_exp, side="left")

    total_pos = gts_batch.sum(dim=1, keepdim=True)
    total_neg = d - total_pos

    tps = total_pos - (pos_idx - total_neg).clamp_min(0)
    fps = total_neg - (neg_idx - total_pos).clamp_min(0)

    fns = total_pos - tps
    tns = total_neg - fps

    row0 = torch.stack([tns, fps], dim=-1)
    row1 = torch.stack([fns, tps], dim=-1)

    return torch.stack([row0, row1], dim=2)


def _get_linspaced_thresholds(anomaly_maps: torch.Tensor, num_thresholds: int) -> torch.Tensor:
    """Get linearly spaced thresholds between min and max anomaly scores.

    Args:
        anomaly_maps: Anomaly score maps
        num_thresholds: Number of thresholds to generate

    Returns:
        Linearly spaced thresholds of shape ``(num_thresholds,)``

    Raises:
        ValueError: If threshold bounds are invalid
    """
    _validate.is_num_thresholds_gte2(num_thresholds)
    # this operation can be a bit expensive
    thresh_low, thresh_high = thresh_bounds = (anomaly_maps.min().item(), anomaly_maps.max().item())
    try:
        _validate.validate_threshold_bounds(thresh_bounds)
    except ValueError as ex:
        msg = f"Invalid threshold bounds computed from the given anomaly maps. Cause: {ex}"
        raise ValueError(msg) from ex
    return torch.linspace(thresh_low, thresh_high, num_thresholds, dtype=anomaly_maps.dtype)


def threshold_and_binary_classification_curve(
    anomaly_maps: torch.Tensor,
    masks: torch.Tensor,
    threshold_choice: ThresholdMethod | str = ThresholdMethod.MINMAX_LINSPACE,
    thresholds: torch.Tensor | None = None,
    num_thresholds: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get thresholds and binary classification matrices for a batch of images.

    Args:
        anomaly_maps: Anomaly score maps of shape ``(N, H, W)``
        masks: Binary ground truth masks of shape ``(N, H, W)``
        threshold_choice: Method for selecting thresholds. Defaults to
            ``MINMAX_LINSPACE``
        thresholds: Sequence of thresholds to use. Only used when
            ``threshold_choice`` is ``GIVEN``
        num_thresholds: Number of thresholds between min and max scores. Only
            used when ``threshold_choice`` is ``MINMAX_LINSPACE``

    Returns:
        Tuple containing:

        - Thresholds of shape ``(K,)`` with same dtype as ``anomaly_maps``
        - Binary classification matrices of shape ``(N, K, 2, 2)`` where:

          - ``[..., 1, 1]``: True Positives (TP)
          - ``[..., 0, 1]``: False Positives (FP)
          - ``[..., 1, 0]``: False Negatives (FN)
          - ``[..., 0, 0]``: True Negatives (TN)

        The counts are per-instance pixel counts. Thresholds are shared across
        instances and sorted in ascending order.

    Example:
        >>> maps = torch.rand(10, 32, 32)  # 10 images
        >>> masks = torch.randint(0, 2, (10, 32, 32)).bool()
        >>> thresh, curves = threshold_and_binary_classification_curve(
        ...     maps,
        ...     masks,
        ...     num_thresholds=10,
        ... )
        >>> thresh.shape, curves.shape
        (torch.Size([10]), torch.Size([10, 10, 2, 2]))
    """
    threshold_choice = ThresholdMethod(threshold_choice)
    _validate.is_anomaly_maps(anomaly_maps)
    _validate.is_masks(masks)
    _validate.is_same_shape(anomaly_maps, masks)

    if threshold_choice == ThresholdMethod.GIVEN:
        assert thresholds is not None
        _validate.is_valid_threshold(thresholds)
        if num_thresholds is not None:
            logger.warning(
                "Argument `num_thresholds` was given, "
                f"but it is ignored because `thresholds_choice` is '{threshold_choice.value}'.",
            )
        thresholds = thresholds.to(anomaly_maps.dtype)

    elif threshold_choice == ThresholdMethod.MINMAX_LINSPACE:
        assert num_thresholds is not None
        if thresholds is not None:
            logger.warning(
                "Argument `thresholds_given` was given, "
                f"but it is ignored because `thresholds_choice` is '{threshold_choice.value}'.",
            )
        # `num_thresholds` is validated in the function below
        thresholds = _get_linspaced_thresholds(anomaly_maps, num_thresholds)

    elif threshold_choice == ThresholdMethod.MEAN_FPR_OPTIMIZED:
        raise NotImplementedError(f"TODO implement {threshold_choice.value}")  # noqa: EM102

    else:
        msg = (
            f"Expected `threshs_choice` to be from {list(ThresholdMethod.__members__)},"
            f" but got '{threshold_choice.value}'"
        )
        raise NotImplementedError(msg)

    # keep the batch dimension and flatten the rest
    scores_batch = anomaly_maps.reshape(anomaly_maps.shape[0], -1)
    gts_batch = masks.reshape(masks.shape[0], -1).to(dtype=torch.bool)

    binclf_curves = binary_classification_curve(scores_batch, gts_batch, thresholds)

    num_images = anomaly_maps.shape[0]

    try:
        _validate.is_binclf_curves(binclf_curves, valid_thresholds=thresholds)

        # these two validations cannot be done in `_validate.binclf_curves`
        # because it does not have access to the original shapes of
        # `anomaly_maps`
        if binclf_curves.shape[0] != num_images:
            msg = (
                "Expected `binclf_curves` to have the same number of images as `anomaly_maps`, "
                f"but got {binclf_curves.shape[0]} and {anomaly_maps.shape[0]}"
            )
            raise RuntimeError(msg)

    except (TypeError, ValueError) as ex:
        msg = f"Invalid `binclf_curves` was computed. Cause: {ex}"
        raise RuntimeError(msg) from ex

    return thresholds, binclf_curves


def per_image_tpr(binclf_curves: torch.Tensor) -> torch.Tensor:
    """Compute True Positive Rate (TPR) for each image at each threshold.

    TPR = TP / P = TP / (TP + FN)

    Where:
        - TP: True Positives
        - FN: False Negatives
        - P: Total Positives (TP + FN)

    Args:
        binclf_curves: Binary classification curves of shape ``(N, K, 2, 2)``
            See :func:`binary_classification_curve`

    Returns:
        TPR values of shape ``(N, K)`` and dtype ``float64`` where:
            - N: number of images
            - K: number of thresholds

        TPR is in descending order since thresholds are sorted ascending.
        TPR will be NaN for normal images (P = 0).

    Example:
        >>> curves = torch.randint(0, 10, (5, 10, 2, 2))  # 5 imgs, 10 thresh
        >>> tpr = per_image_tpr(curves)
        >>> tpr.shape
        torch.Size([5, 10])
    """
    # shape: (num images, num thresholds)
    tps = binclf_curves[..., 1, 1]
    pos = binclf_curves[..., 1, :].sum(dim=2)

    # tprs will be nan if pos == 0 (normal image), which is expected
    return tps.to(torch.float64) / pos.to(torch.float64)


def per_image_fpr(binclf_curves: torch.Tensor) -> torch.Tensor:
    """Compute False Positive Rate (FPR) for each image at each threshold.

    FPR = FP / N = FP / (FP + TN)

    Where:
        - FP: False Positives
        - TN: True Negatives
        - N: Total Negatives (FP + TN)

    Args:
        binclf_curves: Binary classification curves of shape ``(N, K, 2, 2)``
            See :func:`binary_classification_curve`

    Returns:
        FPR values of shape ``(N, K)`` and dtype ``float64`` where:
            - N: number of images
            - K: number of thresholds

        FPR is in descending order since thresholds are sorted ascending.
        FPR will be NaN for fully anomalous images (N = 0).

    Example:
        >>> curves = torch.randint(0, 10, (5, 10, 2, 2))  # 5 imgs, 10 thresh
        >>> fpr = per_image_fpr(curves)
        >>> fpr.shape
        torch.Size([5, 10])
    """
    # shape: (num images, num thresholds)
    fps = binclf_curves[..., 0, 1]
    neg = binclf_curves[..., 0, :].sum(dim=2)

    # it can be `nan` if an anomalous image is fully covered by the mask
    return fps.to(torch.float64) / neg.to(torch.float64)
