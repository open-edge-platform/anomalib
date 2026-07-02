# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SuperADD: PatchCore-style anomaly detection on a DINOv3 backbone.

SuperADD is an anomaly detection model that builds a memory bank of patch
features extracted from a pretrained DINOv3 Vision Transformer. Multi-layer ViT
token features are computed over overlapping image patches, a per-layer memory
bank is built from normal training images via distance-based coreset
subsampling, and test images are scored by nearest-neighbor distance to the
bank.

Example:
    >>> from anomalib.data import MVTecAD2
    >>> from anomalib.models import SuperADD
    >>> from anomalib.engine import Engine

    >>> # Initialize model and data
    >>> datamodule = MVTecAD2()
    >>> model = SuperADD()

    >>> # Train using the Engine
    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)

    >>> # Get predictions
    >>> predictions = engine.predict(model=model, datamodule=datamodule)
"""

from .lightning_model import SuperADD

__all__ = ["SuperADD"]
