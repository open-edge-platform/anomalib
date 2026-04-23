# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AnomalyVFM -- Transforming Vision Foundation Models into Zero-Shot Anomaly Detectors.

AnomalyVFM implements a zero-shot anomaly detector on top of pretrained VFMs.
It does so by first generating synthetic images using FLUX and training on top of them.
The model directly predicts an anomaly score and a anomaly mask.

Example:
    >>> from anomalib.models.image import AnomalyVFM
    >>> model = AnomalyVFM("radio")

The model can be used with any of the supported datasets.

Notes:
    - Is already pretrained, i.e. requires no training
    - Supports 4 VFM backbones (RADIO, SigLIP2, CLIP and DINOv2)
    - Supports both anomaly detection and localization tasks
    - Requires significant GPU memory due to Vision Transformer architecture

See Also:
    :class:`anomalib.models.image.dinomaly.lightning_model.AnomalyVFM`:
        Lightning implementation of the AnomalyVFM model.
"""

from .lightning_model import AnomalyVFM

__all__ = ["AnomalyVFM"]
