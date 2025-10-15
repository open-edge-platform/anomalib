# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Post-processing module for MEBin-based anomaly detection results.

This module provides post-processing functionality for anomaly detection
outputs through the :class:`MEBinPostProcessor` class.

MEBin was introduced in AnomalyNCD : https://arxiv.org/pdf/2410.14379

The MEBin post-processor handles:
    - Converting anomaly maps to binary masks using MEBin algorithm
    - Sampling anomaly maps at configurable rates for efficient processing
    - Applying morphological operations (erosion) to refine binary masks
    - Maintaining minimum interval lengths for consistent mask generation
    - Formatting results for downstream use

Example:
    >>> from anomalib.post_processing import MEBinPostProcessor
    >>> post_processor = MEBinPostProcessor(sample_rate=4, min_interval_len=4)
    >>> predictions = post_processor(anomaly_maps=anomaly_maps)
"""

import numpy as np
import torch

from anomalib.data import InferenceBatch
from anomalib.metrics import MEBin

from .post_processor import PostProcessor


class MEBinPostProcessor(PostProcessor):
    """Post-processor for MEBin-based anomaly detection.

    This class handles post-processing of anomaly detection results by:
        - Converting continuous anomaly maps to binary masks using MEBin algorithm
        - Sampling anomaly maps at configurable rates for efficient processing
        - Applying morphological operations (erosion) to refine binary masks
        - Maintaining minimum interval lengths for consistent mask generation
        - Formatting results for downstream use

    Args:
        sample_rate (int): Threshold sampling step size.
            Defaults to 4
        min_interval_len (int): Minimum length of the stable interval. Can be adjusted based on the interval
            between normal and abnormal score distributions in the anomaly score maps.
            Decrease if there are many false negatives, increase if there are many false positives.
            Defaults to 4
        erode (bool): Whether to perform erosion after binarization to eliminate noise.
            Defaults to True
        **kwargs: Additional keyword arguments passed to parent class.

    Example:
        >>> from anomalib.post_processing import MEBinPostProcessor
        >>> post_processor = MEBinPostProcessor(sample_rate=4, min_interval_len=4)
        >>> predictions = post_processor(anomaly_maps=anomaly_maps)
    """

    def __init__(
        self,
        sample_rate: int = 4,
        min_interval_len: int = 4,
        erode: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.sample_rate = sample_rate
        self.min_interval_len = min_interval_len
        self.erode = erode

    """Custom post-processor using MEBin algorithm"""

    def forward(self, predictions: InferenceBatch) -> InferenceBatch:
        """Post-process model predictions using MEBin algorithm.

        This method converts continuous anomaly maps to binary masks using the MEBin
        algorithm, which provides efficient and accurate binarization of anomaly
        detection results.

        Args:
            predictions (InferenceBatch): Batch containing model predictions with
                anomaly maps to be processed.

        Returns:
            InferenceBatch: Post-processed batch with binary masks generated from
                anomaly maps using MEBin algorithm.

        Note:
            The method automatically handles tensor-to-numpy conversion and back,
            ensuring compatibility with the original tensor device and dtype.
        """
        if predictions.anomaly_map is None:
            msg = "Anomaly map is required for MEBin post-processing"
            raise ValueError(msg)

        # Store the original tensor for device and dtype info
        original_anomaly_map = predictions.anomaly_map
        anomaly_maps = original_anomaly_map.detach().cpu().numpy()
        if anomaly_maps.ndim == 4:
            anomaly_maps = anomaly_maps[:, 0, :, :]  # Remove channel dimension

        # Normalize to 0-255 and convert to uint8
        norm_maps = []
        for amap in anomaly_maps:
            amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8) * 255
            norm_maps.append(amap_norm.astype(np.uint8))

        mebin = MEBin(
            anomaly_map_list=norm_maps,
            sample_rate=self.sample_rate,
            min_interval_len=self.min_interval_len,
            erode=self.erode,
        )
        binarized_maps, _ = mebin.binarize_anomaly_maps()

        # Convert back to torch.Tensor and normalize to 0/1
        pred_masks = torch.stack([torch.from_numpy(bm).to(original_anomaly_map.device) for bm in binarized_maps])
        pred_masks = (pred_masks > 0).to(original_anomaly_map.dtype)

        return InferenceBatch(
            pred_label=predictions.pred_label,
            pred_score=predictions.pred_score,
            pred_mask=pred_masks,
            anomaly_map=predictions.anomaly_map,
        )
