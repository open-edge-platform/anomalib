# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""INP-Former: Anomaly Detection via Intrinsic Normal Prototypes.

The INP-Former model implements an encoder-decoder architecture for anomaly detection
that extracts Intrinsic Normal Prototypes (INPs) directly from each test image, rather
than relying on prototypes stored from the training set. Features from a frozen
pre-trained Vision Transformer encoder are aggregated by an INP Extractor (cross-attention
with learnable query tokens) into a small set of INPs per image, fused through a
bottleneck, and reconstructed by an INP-Guided Decoder that uses the INPs as keys and
values to constrain its output to normal patterns.

Anomaly detection is performed by computing the per-token discrepancy between encoder
and decoder features at multiple scales. Because the INPs are derived from the test
image itself, the normality reference is naturally aligned with the input, making the
model effective across single-class, multi-class, and few-shot anomaly detection
settings.

Example:
    >>> from anomalib.models.image import InpFormer
    >>> model = InpFormer()

The model can be used with any of the supported datasets and task modes in
anomalib. It combines pre-trained Vision Transformer features with image-specific
prototype extraction for robust, well-aligned anomaly detection.

Notes:
    - Uses a frozen pre-trained Vision Transformer as the backbone encoder
    - Intrinsic Normal Prototypes are extracted dynamically from each image
    - INP Coherence Loss ensures prototypes faithfully represent normal features
    - Soft Mining Loss upweights hard-to-reconstruct tokens during training
    - Supports both anomaly detection and localization tasks
    - Requires significant GPU memory due to Vision Transformer architecture

See Also:
    :class:`anomalib.models.image.inp_former.lightning_model.InpFormer`:
        Lightning implementation of the INP-Former model.
"""

from anomalib.models.image.inp_former.lightning_model import InpFormer

__all__ = ["InpFormer"]
