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
    Example:
        >>> from anomalib.post_processing import MEBinPostProcessor
        >>> from anomalib.data import InferenceBatch
        >>> import torch
        >>> # Create sample anomaly maps
        >>> anomaly_maps = torch.rand(4, 1, 256, 256)
        >>> predictions = InferenceBatch(anomaly_map=anomaly_maps)
        >>> post_processor = MEBinPostProcessor(sample_rate=4, min_interval_len=4)
        >>> results = post_processor(predictions)
"""

import numpy as np
import torch
from lightning import LightningModule, Trainer

from anomalib.data import Batch, InferenceBatch
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
        >>> from anomalib.data import InferenceBatch
        >>> import torch
        >>> # Create sample predictions
        >>> anomaly_maps = torch.rand(4, 1, 256, 256)
        >>> predictions = InferenceBatch(anomaly_map=anomaly_maps)
        >>> post_processor = MEBinPostProcessor(sample_rate=4, min_interval_len=4)
        >>> results = post_processor(predictions)
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

        # Convert to proper format for MEBin (don't normalize individually)
        # MEBin will handle normalization after determining the global min/max range
        anomaly_maps_list = [amap.astype(np.float32) for amap in anomaly_maps]

        mebin = MEBin(
            anomaly_map_list=anomaly_maps_list,
            sample_rate=self.sample_rate,
            min_interval_len=self.min_interval_len,
            erode=self.erode,
        )
        binarized_maps, _ = mebin.binarize_anomaly_maps()

        # Convert back to torch.Tensor and normalize to 0/1
        pred_masks = torch.stack([torch.from_numpy(bm).to(original_anomaly_map.device) for bm in binarized_maps])
        pred_masks = (pred_masks > 0).to(original_anomaly_map.dtype)

        # Create result with MEBin pred_mask
        result = InferenceBatch(
            pred_label=predictions.pred_label,
            pred_score=predictions.pred_score,
            pred_mask=pred_masks,
            anomaly_map=predictions.anomaly_map,
        )

        # Apply parent class post-processing for normalization and thresholding
        # This will compute pred_label from pred_score if needed
        return super().forward(result)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Batch,
        *args,
        **kwargs,
    ) -> None:
        """Apply MEBin post-processing to test batch predictions.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance.
            pl_module (LightningModule): PyTorch Lightning module instance.
            outputs (Batch): Batch containing model predictions.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        del trainer, pl_module, args, kwargs  # Unused arguments
        self.post_process_batch(outputs)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Batch,
        *args,
        **kwargs,
    ) -> None:
        """Apply MEBin post-processing to prediction batch.

        Args:
            trainer (Trainer): PyTorch Lightning trainer instance.
            pl_module (LightningModule): PyTorch Lightning module instance.
            outputs (Batch): Batch containing model predictions.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        del trainer, pl_module, args, kwargs  # Unused arguments
        self.post_process_batch(outputs)

    def post_process_batch(self, batch: Batch) -> None:
        """Post-process a batch of predictions using MEBin algorithm.

        This method applies MEBin binarization to anomaly maps in the batch and
        updates the pred_mask field with the binarized results.

        Args:
            batch (Batch): Batch containing model predictions to be processed.
        """
        if batch.anomaly_map is None:
            return

        # Store the original tensor for device and dtype info
        original_anomaly_map = batch.anomaly_map
        anomaly_maps = original_anomaly_map.detach().cpu().numpy()

        # Handle different tensor shapes
        if anomaly_maps.ndim == 4:
            anomaly_maps = anomaly_maps[:, 0, :, :]  # Remove channel dimension if present
        elif anomaly_maps.ndim == 3:
            # Already in correct format (batch, height, width)
            pass
        else:
            msg = f"Unsupported anomaly map shape: {anomaly_maps.shape}"
            raise ValueError(msg)

        # Convert to proper format for MEBin (don't normalize individually)
        # MEBin will handle normalization after determining the global min/max range
        anomaly_maps_list = [amap.astype(np.float32) for amap in anomaly_maps]

        # Apply MEBin binarization
        mebin = MEBin(
            anomaly_map_list=anomaly_maps_list,
            sample_rate=self.sample_rate,
            min_interval_len=self.min_interval_len,
            erode=self.erode,
        )
        binarized_maps, _ = mebin.binarize_anomaly_maps()

        # Convert back to torch.Tensor and normalize to 0/1
        pred_masks = torch.stack([torch.from_numpy(bm).to(original_anomaly_map.device) for bm in binarized_maps])
        pred_masks = (pred_masks > 0).to(original_anomaly_map.dtype)

        # Add channel dimension if original had one
        if original_anomaly_map.ndim == 4:
            pred_masks = pred_masks.unsqueeze(1)

        # Update the batch with binarized masks
        batch.pred_mask = pred_masks

        # Apply parent class post-processing for normalization and thresholding
        # This will compute pred_label from pred_score if needed
        super().post_process_batch(batch)
