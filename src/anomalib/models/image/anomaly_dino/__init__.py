# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2.

This module implements AnomalyDINO. A memory-bank model for anomaly detection
that utilizes DINOv2-Small as its backbone. At inference time it uses kNN
to search for anomalous patches. The image anomaly score is dependent on the worst
99th percentile of the pixel-wise anomaly score.

The model has optional masking to remove noisy background components,
also optionally can use greedy coreset-subsampling if needed.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models.image.anomaly_dino.lightning_model import AnomalyDINO
    >>> from anomalib.engine import Engine

    >>> MVTEC_CATEGORIES = [
    ...     "hazelnut", "grid", "carpet", "bottle", "cable", "capsule", "leather",
    ...     "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"
    ... ]
    >>> MASKED_CATEGORIES = ["capsule", "hazelnut", "pill", "screw", "toothbrush"]

    >>> for category in MVTEC_CATEGORIES:
    ...     mask = category in MASKED_CATEGORIES
    ...     print(f"--- Running category: {category} | masking={mask} ---")

    ...     # Initialize data module
    ...     datamodule = MVTecAD(category=category)

    ...     # Initialize model
    ...     model = AnomalyDINO(
    ...         num_neighbours=1,
    ...         encoder_name="dinov2_vit_small_14",
    ...         masking=mask,
    ...         coreset_subsampling=False,
    ...     )

    ...     # Train and test
    ...     engine = Engine()
    ...     engine.fit(model=model, datamodule=datamodule)
    ...     engine.test(datamodule=datamodule)
    >>> print("All categories processed.")
"""

from anomalib.models.image.anomaly_dino.lightning_model import AnomalyDINO

__all__ = ["AnomalyDINO"]
