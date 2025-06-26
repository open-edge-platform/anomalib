# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Core Components for the Tiled Ensemble Pipeline.

This module aggregates the essential components utilized in the Tiled Ensemble.
Each component is designed to handle specific stages of the pipeline.

Included Components:
    - `merging`: Job to merge predictions from individual tiles into a whole output.
    - `metrics_calculation`: Job to compute evaluation metrics at image levels.
    - `model_training`: Job for training models on individual tiles.
    - `normalization`: Job to normalize predictions.
    - `prediction`: Job to generate predictions for each tile.
    - `smoothing`: Job to smooth seams between tiles for seamless reconstruction.
    - `stats_calculation`: Job to compute statistics required by normalisation and thresholding.
    - `thresholding`: Job to apply thresholds to predictions.
    - `visualization`: Job to visualize predictions and anomalies.

Reference:
    Blaž Rolih, Dick Ameln, Ashwin Vaidya, Samet Akçay:
    "Divide and Conquer: High-Resolution Industrial Anomaly Detection via
    Memory Efficient Tiled Ensemble." Proceedings of the IEEE/CVF Conference
    on Computer Vision and Pattern Recognition Workshops (VAND 2.0). 2024.
"""

from .merging import MergeJobGenerator
from .metrics_calculation import MetricsCalculationJobGenerator
from .model_training import TrainModelJobGenerator
from .normalization import NormalizationJobGenerator
from .prediction import PredictJobGenerator
from .smoothing import SmoothingJobGenerator
from .stats_calculation import StatisticsJobGenerator
from .thresholding import ThresholdingJobGenerator
from .utils import NormalizationStage, PredictData, ThresholdingStage
from .visualization import VisualizationJobGenerator

__all__ = [
    "NormalizationStage",
    "ThresholdingStage",
    "PredictData",
    "TrainModelJobGenerator",
    "PredictJobGenerator",
    "MergeJobGenerator",
    "SmoothingJobGenerator",
    "StatisticsJobGenerator",
    "NormalizationJobGenerator",
    "ThresholdingJobGenerator",
    "VisualizationJobGenerator",
    "MetricsCalculationJobGenerator",
]
