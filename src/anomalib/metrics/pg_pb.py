# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PGn and PBn metrics for binary image-level classification tasks.

This module provides two metrics for evaluating binary image-level classification performance
on the assumption that bad (anomalous) samples are considered to be the positive class:

- ``PGn``: Presorted good with n% bad samples missed, can be interpreted as true negative rate
at a fixed false negative rate (TNR@nFNR).
- ``PBn``: Presorted bad with n% good samples misclassified, can be interpreted as true positive rate
at a fixed false positive rate (TPR@nFPR).

These metrics emphasize the practical applications of anomaly detection models by showing their potential
to reduce human operator workload while maintaining an acceptable level of misclassification.

Example:
    >>> from anomalib.metrics import PGn, PBn
    >>> from anomalib.data import ImageBatch
    >>> import torch
    >>> # Create sample batch
    >>> batch = ImageBatch(
    ...     image=torch.rand(4, 3, 32, 32),
    ...     pred_score=torch.tensor([0.1, 0.4, 0.35, 0.8]),
    ...     gt_label=torch.tensor([0, 0, 1, 1])
    ... )
    >>> pg = PGn(fnr=0.2)
    >>> # Print name of the metric
    >>> print(pg.name)
    PG20
    >>> # Compute PGn score
    >>> pg.update(batch)
    >>> pg.compute()
    tensor(1.0)
    >>> pb = PBn(fpr=0.2)
    >>> # Print name of the metric
    >>> print(pb.name)
    PB20
    >>> # Compute PBn score
    >>> pb.update(batch)
    >>> pb.compute()
    tensor(1.0)

Note:
    Scores for both metrics range from 0 to 1, with 1 indicating perfect separation
    of the respective class with ``n``% or less of the other class misclassified.

Reference:
    Aimira Baitieva, Yacine Bouaouni, Alexandre Briot, Dick Ameln, Souhaiel Khalfaoui,
    Samet Akcay; Beyond Academic Benchmarks: Critical Analysis and Best Practices
    for Visual Industrial Anomaly Detection; in: Proceedings of the IEEE/CVF Conference
    on Computer Vision and Pattern Recognition (CVPR) Workshops, 2025, pp. 4024-4034,
    https://arxiv.org/abs/2503.23451
"""

import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat

from anomalib.metrics.base import AnomalibMetric


class _PGn(Metric):
    """Presorted good metric.

    This class calculates the Presorted good (PGn) metric, which is the true negative rate
    at a fixed false negative rate.

    Args:
        **kwargs: Additional arguments passed to the parent ``Metric`` class.

    Attributes:
        fnr (torch.Tensor): Fixed false negative rate (bad parts misclassified).
        Defaults to ``0.05``.

    Example:
        >>> from anomalib.metrics.pg_pb import _PGn
        >>> import torch
        >>> # Create sample data
        >>> preds = torch.tensor([0.1, 0.4, 0.35, 0.8])
        >>> target = torch.tensor([0, 0, 1, 1])
        >>> # Compute PGn score
        >>> pg = _PGn(fnr=0.2)
        >>> pg.update(preds, target)
        >>> pg.compute()
        tensor(1.0)
    """

    def __init__(self, fnr: float = 0.05, **kwargs) -> None:
        super().__init__(**kwargs)
        if fnr < 0 or fnr > 1:
            msg = f"False negative rate must be in the range between 0 and 1, got {fnr}."
            raise ValueError(msg)

        self.fnr = torch.tensor(fnr, dtype=torch.float32)
        self.name = "PG" + str(int(fnr * 100))

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with new values.

        Args:
            preds (torch.Tensor): predictions of the model
            target (torch.Tensor): ground truth targets
        """
        self.target.append(target)
        self.preds.append(preds)

    def compute(self) -> torch.Tensor:
        """Compute the PGn score at a given false negative rate.

        Returns:
            torch.Tensor: PGn score value.

        Raises:
            ValueError: If no negative samples are found.
        """
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)

        pos_scores = preds[target == 1]
        thr_accept = torch.quantile(pos_scores, self.fnr)

        neg_scores = preds[target == 0]
        if neg_scores.numel() == 0:
            msg = "No negative samples found. Cannot compute PGn score."
            raise ValueError(msg)
        pg = neg_scores[neg_scores < thr_accept].numel() / neg_scores.numel()

        return torch.tensor(pg, dtype=preds.dtype)


class PGn(AnomalibMetric, _PGn):  # type: ignore[misc]
    """Wrapper to add AnomalibMetric functionality to PGn metric.

    This class wraps the internal ``_PGn`` metric to make it compatible with
    Anomalib's batch processing capabilities.
    """

    default_fields = ("pred_score", "gt_label")


class _PBn(Metric):
    """Presorted bad metric.

    This class calculates the Presorted bad (PBn) metric, which is the true positive rate
    at a fixed false positive rate.

    Args:
        fpr (float): Fixed false positive rate (good parts misclassified). Defaults to ``0.05``.
        **kwargs: Additional arguments passed to the parent ``Metric`` class.

    Example:
        >>> from anomalib.metrics import _PBn
        >>> import torch
        >>> preds = torch.tensor([0.1, 0.4, 0.35, 0.8])
        >>> target = torch.tensor([0, 0, 1, 1])
        >>> pb = _PBn(fpr=0.2)
        >>> pb.update(preds, target)
        >>> pb.compute()
        tensor(1.0)
    """

    def __init__(self, fpr: float = 0.05, **kwargs) -> None:
        super().__init__(**kwargs)
        if fpr < 0 or fpr > 1:
            msg = f"False positive rate must be in the range between 0 and 1, got {fpr}."
            raise ValueError(msg)

        self.fpr = torch.tensor(fpr, dtype=torch.float32)
        self.name = "PB" + str(int(fpr * 100))

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with new values.

        Args:
            preds (torch.Tensor): predictions of the model
            target (torch.Tensor): ground truth targets
        """
        self.target.append(target)
        self.preds.append(preds)

    def compute(self) -> torch.Tensor:
        """Compute the PBn score at a given false positive rate.

        Returns:
            torch.Tensor: PBn score value.

        Raises:
            ValueError: If no positive samples are found.
        """
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)

        neg_scores = preds[target == 0]
        thr_accept = torch.quantile(neg_scores, 1 - self.fpr)

        pos_scores = preds[target == 1]
        if pos_scores.numel() == 0:
            msg = "No positive samples found. Cannot compute PBn score."
            raise ValueError(msg)
        pb = pos_scores[pos_scores > thr_accept].numel() / pos_scores.numel()

        return torch.tensor(pb, dtype=preds.dtype)


class PBn(AnomalibMetric, _PBn):  # type: ignore[misc]
    """Wrapper to add AnomalibMetric functionality to PBn metric.

    This class wraps the internal ``_PBn`` metric to make it compatible with
    Anomalib's batch processing capabilities.
    """

    default_fields = ("pred_score", "gt_label")
