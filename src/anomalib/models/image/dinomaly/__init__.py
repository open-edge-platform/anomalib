# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dinomaly: Vision Transformer-based Anomaly Detection with Feature Reconstruction.

The Dinomaly model implements a Vision Transformer encoder-decoder architecture for
anomaly detection using pre-trained DINOv2 features. The model extracts features from
multiple intermediate layers of a DINOv2 encoder, compresses them through a bottleneck
MLP, and reconstructs them using a Vision Transformer decoder.

Anomaly detection is performed by computing cosine similarity between encoder and decoder
features at multiple scales. The model is particularly effective for visual anomaly
detection tasks where the goal is to identify regions or images that deviate from
normal patterns learned during training.

Example:
    >>> from anomalib.models.image import Dinomaly
    >>> model = Dinomaly()

The model can be used with any of the supported datasets and task modes in
anomalib. It leverages the powerful feature representations from DINOv2 Vision
Transformers combined with a reconstruction-based approach for robust anomaly detection.

Notes:
    - Uses DINOv2 Vision Transformer as the backbone encoder
    - Features are extracted from intermediate layers for multi-scale analysis
    - Employs feature reconstruction loss for unsupervised learning
    - Supports both anomaly detection and localization tasks
    - Requires significant GPU memory due to Vision Transformer architecture

See Also:
    :class:`anomalib.models.image.dinomaly.lightning_model.Dinomaly`:
        Lightning implementation of the Dinomaly model.
"""

from anomalib.models.image.dinomaly.lightning_model import Dinomaly

__all__ = ["Dinomaly"]
