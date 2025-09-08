# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom metrics for evaluating anomaly detection models.

This module provides various metrics for evaluating anomaly detection performance:

- Area Under Curve (AUC) metrics:
    - ``AUROC``: Area Under Receiver Operating Characteristic curve
    - ``AUPR``: Area Under Precision-Recall curve
    - ``AUPRO``: Area Under Per-Region Overlap curve
    - ``AUPIMO``: Area Under Per-Image Missed Overlap curve

- F1-score metrics:
    - ``F1Score``: Standard F1 score
    - ``F1Max``: Maximum F1 score across thresholds

- Threshold metrics:
    - ``F1AdaptiveThreshold``: Finds optimal threshold by maximizing F1 score
    - ``ManualThreshold``: Uses manually specified threshold

- Other metrics:
    - ``AnomalibMetric``: Base class for custom metrics
    - ``AnomalyScoreDistribution``: Analyzes score distributions
    - ``BinaryPrecisionRecallCurve``: Computes precision-recall curves
    - ``Evaluator``: Combines multiple metrics for evaluation
    - ``MinMax``: Normalizes scores to [0,1] range
    - ``PBn``: Presorted bad with n% good samples misclassified
    - ``PGn``: Presorted good with n% bad samples missed
    - ``PRO``: Per-Region Overlap score
    - ``PIMO``: Per-Image Missed Overlap score

Example:
    >>> from anomalib.metrics import AUROC, F1Score
    >>> from anomalib.data import ImageBatch
    >>> import torch
    >>> # Initialize metrics with required fields
    >>> auroc = AUROC(fields=["pred_score", "gt_label"])
    >>> f1 = F1Score(fields=["pred_score", "gt_label"], threshold=0.5)
    >>> # Create sample batch
    >>> batch = ImageBatch(
    ...     image=torch.rand(4, 3, 32, 32),
    ...     pred_score=torch.tensor([0.1, 0.9, 0.2, 0.8]),
    ...     gt_label=torch.tensor([0, 1, 0, 1])
    ... )
    >>> # Calculate metrics
    >>> auroc(batch)
    tensor(1.)
    >>> f1(batch)
    tensor(1.)
"""

from .anomaly_score_distribution import AnomalyScoreDistribution
from .aupr import AUPR
from .aupro import AUPRO
from .auroc import AUROC
from .base import AnomalibMetric, create_anomalib_metric
from .evaluator import Evaluator
from .f1_score import F1Max, F1Score
from .min_max import MinMax
from .pg_pb import PBn, PGn
from .pimo import AUPIMO, PIMO
from .precision_recall_curve import BinaryPrecisionRecallCurve
from .pro import PRO
from .threshold import F1AdaptiveThreshold, ManualThreshold

__all__ = [
    "AUROC",
    "AUPR",
    "AUPRO",
    "AnomalibMetric",
    "AnomalyScoreDistribution",
    "BinaryPrecisionRecallCurve",
    "create_anomalib_metric",
    "Evaluator",
    "F1AdaptiveThreshold",
    "F1Max",
    "F1Score",
    "ManualThreshold",
    "MinMax",
    "PGn",
    "PBn",
    "PRO",
    "PIMO",
    "AUPIMO",
]
