# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from anomalib.data import NumpyImageBatch as PredictionResult


@dataclass(kw_only=True)
class InferenceData:
    """InferenceData represents the data that is produced by the inference stage of the pipeline."""

    prediction: PredictionResult  # prediction result, e.g., bounding boxes, masks, etc.
    visualized_prediction: np.ndarray  # visualized prediction (e.g., bounding boxes, masks, etc. drawn on the frame)
    model_name: str  # name of the model that produced the prediction


@dataclass(kw_only=True)
class StreamData:
    """StreamData represents the data that flows through the various stages of the pipeline.
    Each stage of the pipeline may set some of the attributes of this class,
    making the corresponding information available to the subsequent stages.
    """

    # available after 'stream loading' stage
    frame_data: np.ndarray  # frame loaded as numpy array
    timestamp: float  # timestamp of the frame (epoch)
    source_metadata: dict[str, Any]  # unstructured metadata about the source of the frame (camera ID, video file, etc.)

    # available after 'data monitoring' stage
    # TODO add information such as frame quality, anomalies detected, etc.

    # available after 'inference' stage
    inference_data: InferenceData | None = None

    # available after 'model monitoring' stage
    # TODO add information about drift detection, anomalies, etc.
