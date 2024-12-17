"""F1 adaptive threshold metric for anomaly detection.

This module provides the ``F1AdaptiveThreshold`` class which automatically finds
the optimal threshold value by maximizing the F1 score on validation data.

The threshold is computed by:
1. Computing precision-recall curve across multiple thresholds
2. Calculating F1 score at each threshold point
3. Selecting threshold that yields maximum F1 score

Example:
    >>> from anomalib.metrics import F1AdaptiveThreshold
    >>> import torch
    >>> # Create sample data
    >>> labels = torch.tensor([0, 0, 0, 1, 1])  # Binary labels
    >>> scores = torch.tensor([2.3, 1.6, 2.6, 7.9, 3.3])  # Anomaly scores
    >>> # Initialize and compute threshold
    >>> threshold = F1AdaptiveThreshold(default_value=0.5)
    >>> optimal_threshold = threshold(scores, labels)
    >>> optimal_threshold
    tensor(3.3000)

Note:
    The validation set should contain both normal and anomalous samples for
    reliable threshold computation. A warning is logged if no anomalous samples
    are found.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch

from anomalib.metrics.precision_recall_curve import BinaryPrecisionRecallCurve

from .base import Threshold

logger = logging.getLogger(__name__)


class F1AdaptiveThreshold(BinaryPrecisionRecallCurve, Threshold):
    """Adaptive threshold that maximizes F1 score.

    This class computes and stores the optimal threshold for converting anomaly
    scores to binary predictions by maximizing the F1 score on validation data.

    Args:
        default_value: Initial threshold value used before computation.
            Defaults to ``0.5``.
        **kwargs: Additional arguments passed to parent classes.

    Attributes:
        value (torch.Tensor): Current threshold value.

    Example:
        >>> from anomalib.metrics import F1AdaptiveThreshold
        >>> import torch
        >>> # Create validation data
        >>> labels = torch.tensor([0, 0, 1, 1])  # 2 normal, 2 anomalous
        >>> scores = torch.tensor([0.1, 0.2, 0.8, 0.9])  # Anomaly scores
        >>> # Initialize threshold
        >>> threshold = F1AdaptiveThreshold()
        >>> # Compute optimal threshold
        >>> optimal_value = threshold(scores, labels)
        >>> print(f"Optimal threshold: {optimal_value:.4f}")
        Optimal threshold: 0.5000
    """

    def __init__(self, default_value: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_state("value", default=torch.tensor(default_value), persistent=True)
        self.value = torch.tensor(default_value)

    def compute(self) -> torch.Tensor:
        """Compute optimal threshold by maximizing F1 score.

        Calculates precision-recall curve and corresponding thresholds, then
        finds the threshold that maximizes the F1 score.

        Returns:
            torch.Tensor: Optimal threshold value.

        Warning:
            If validation set contains no anomalous samples, the threshold will
            default to the maximum anomaly score, which may lead to poor
            performance.
        """
        precision: torch.Tensor
        recall: torch.Tensor
        thresholds: torch.Tensor

        if not any(1 in batch for batch in self.target):
            msg = (
                "The validation set does not contain any anomalous images. As a "
                "result, the adaptive threshold will take the value of the "
                "highest anomaly score observed in the normal validation images, "
                "which may lead to poor predictions. For a more reliable "
                "adaptive threshold computation, please add some anomalous "
                "images to the validation set."
            )
            logging.warning(msg)

        precision, recall, thresholds = super().compute()
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
        if thresholds.dim() == 0:
            # special case where recall is 1.0 even for the highest threshold.
            # In this case 'thresholds' will be scalar.
            self.value = thresholds
        else:
            self.value = thresholds[torch.argmax(f1_score)]
        return self.value

    def __repr__(self) -> str:
        """Return string representation including current threshold value.

        Returns:
            str: String in format "ClassName(value=X.XX)"
        """
        return f"{super().__repr__()} (value={self.value:.2f})"
