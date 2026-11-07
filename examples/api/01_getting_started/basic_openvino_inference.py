# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Getting Started with Anomalib Inference using the Python API.

This example shows how to perform inference on a trained model
using the Anomalib Python API.
"""

# 1. Import required modules
from anomalib.deploy import OpenVINOInferencer

# 2. Initialize the inferencer
inferencer = OpenVINOInferencer(
    path="/path/to/openvino/model.bin",
)

# 4. Get predictions
predictions = inferencer.predict(
    image="/path/to/image.png",
)

# 5. Access the results
if predictions is not None:
    for prediction in predictions:
        anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
        pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
        pred_score = prediction.pred_score  # Image-level anomaly score
