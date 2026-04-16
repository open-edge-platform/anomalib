# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiled Ensemble Pipeline for High-Resolution Anomaly Detection.

This module implements a memory-efficient tiled ensemble approach for
anomaly detection, enabling high-resolution image processing without exceeding GPU memory constraints.
The pipeline divides input images into a grid of tiles, training a dedicated model for each tile location.
By introducing overlapping tiles, it leverages the benefits of traditional
stacking ensembles, enhancing anomaly detection capabilities beyond high resolution alone.

Configuration:
    The pipeline is configured via the `ens_config.yaml` file, which
    includes settings for:
        - Tile size and stride
        - Normalization and thresholding stages
        - Data parameters
        - Seam smoothing parameters
        - Training parameters (e.g., model type, trainer settings)

Usage:
    Training:
        python tools/tiled_ensemble/train_ensemble.py \
            --config tools/tiled_ensemble/ens_config.yaml

    Evaluation:
        python tools/tiled_ensemble/eval.py \
            --config tools/tiled_ensemble/ens_config.yaml \
            --root path_to_results_dir (e.g. results/Padim/MVTec/bottle/v0)

Reference:
    Blaž Rolih, Dick Ameln, Ashwin Vaidya, Samet Akçay:
    "Divide and Conquer: High-Resolution Industrial Anomaly Detection via
    Memory Efficient Tiled Ensemble." Proceedings of the IEEE/CVF Conference
    on Computer Vision and Pattern Recognition Workshops (VAND 2.0). 2024.
"""

from .test_pipeline import EvalTiledEnsemble
from .train_pipeline import TrainTiledEnsemble

__all__ = [
    "TrainTiledEnsemble",
    "EvalTiledEnsemble",
]
