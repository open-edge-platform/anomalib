"""Vision Language Model (VLM) based Anomaly Detection.

This module implements anomaly detection using Vision Language Models (VLMs) like
GPT-4V, LLaVA, etc. The models use natural language prompting to detect anomalies
in images by comparing them with reference normal images.

Example:
    >>> from anomalib.models.image import VlmAd
    >>> model = VlmAd(  # doctest: +SKIP
    ...     backend="chatgpt",
    ...     model_name="gpt-4-vision-preview"
    ... )
    >>> model.fit(["normal1.jpg", "normal2.jpg"])  # doctest: +SKIP
    >>> prediction = model.predict("test.jpg")  # doctest: +SKIP

See Also:
    - :class:`VlmAd`: Main model class for VLM-based anomaly detection
    - :mod:`.backends`: Different VLM backend implementations
    - :mod:`.utils`: Utility functions for prompting and responses
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import VlmAd

__all__ = ["VlmAd"]
