import numpy as np
import torch
import torch.nn.functional as F

from anomalib.data import Batch, InferenceBatch
from anomalib.metrics import MEBin, MinMax

from .post_processor import PostProcessor

class MEBinPostProcessor(PostProcessor):
    """Post-processor for MEBin-based anomaly detection.

    Args:
        sample_rate (int, optional): Threshold sampling step size. Default to 4.
        min_interval_len (int, optional): Minimum length of the stable interval. Default to 4.
        erode (bool, optional): Whether to perform erosion after binarization. Default to True.
        **kwargs: Additional keyword arguments passed to parent class.
    """

    def __init__(
        self,
        sample_rate: int = 4,
        min_interval_len: int = 4,
        erode: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(enable_normalization=True, **kwargs)

        self.sample_rate = sample_rate
        self.min_interval_len = min_interval_len
        self.erode = erode

    @staticmethod
    def _normalize_with_batch_thresholds(
        preds: torch.Tensor | None,
        norm_min: torch.Tensor,
        norm_max: torch.Tensor,
        thresholds: torch.Tensor,
    ) -> torch.Tensor | None:
        """Normalize a tensor using min, max, and batch thresholds.

        Args:
            preds (torch.Tensor | None): Predictions to normalize.
            norm_min (torch.Tensor): Minimum value for normalization.
            norm_max (torch.Tensor): Maximum value for normalization.
            thresholds (torch.Tensor): Threshold values for each sample in batch, shape (B,).

        Returns:
            torch.Tensor | None: Normalized predictions or None if input is None.
        """
        if preds is None or norm_min.isnan() or norm_max.isnan():
            return preds
        
        nan_mask = thresholds.isnan()
        if nan_mask.any():
            default_threshold = (norm_max + norm_min) / 2
            thresholds = torch.where(nan_mask, default_threshold, thresholds)
        
        if preds.dim() == 1:
            preds = ((preds - thresholds) / (norm_max - norm_min)) + 0.5
        else:
            while thresholds.dim() < preds.dim():
                thresholds = thresholds.unsqueeze(-1)
            preds = ((preds - thresholds) / (norm_max - norm_min)) + 0.5
        
        return preds.clamp(min=0, max=1)

    def forward(self, predictions: InferenceBatch) -> InferenceBatch:
        anomaly_maps = predictions.anomaly_map

        if not self.pixel_min.isnan() and not self.pixel_max.isnan():
            normalized_map = (anomaly_maps - self.pixel_min) / (self.pixel_max - self.pixel_min + 1e-8)
            normalized_map = (normalized_map * 255).cpu().numpy().astype(np.uint8)
        else:
            normalized_map = anomaly_maps.cpu().numpy().astype(np.uint8)
        normalized_maps = normalized_map.squeeze(1)
        
        mebin = MEBin(
            anomaly_map_list=normalized_maps,
            sample_rate=self.sample_rate,
            min_interval_len=self.min_interval_len,
            erode=self.erode,
        )
        _, thresholds = mebin.binarize_anomaly_maps()
        thresholds = torch.tensor(thresholds, device=predictions.anomaly_map.device, dtype=predictions.anomaly_map.dtype)
        
        if predictions.pred_score is None and predictions.anomaly_map is None:
            msg = "At least one of pred_score or anomaly_map must be provided."
            raise ValueError(msg)
        pred_score = (
            predictions.pred_score
            if predictions.pred_score is not None
            else torch.amax(anomaly_maps, dim=(-2, -1))
        )

        if self.enable_normalization:
            if not self.pixel_min.isnan() and not self.pixel_max.isnan():
                thresholds_raw = (thresholds / 255.0) * (self.pixel_max - self.pixel_min) + self.pixel_min
            else:
                thresholds_raw = thresholds
            if not self.image_min.isnan() and not self.image_max.isnan():
                thresholds_for_image = (thresholds / 255.0) * (self.image_max - self.image_min) + self.image_min
            else:
                thresholds_for_image = thresholds_raw
            
            pred_score = self._normalize_with_batch_thresholds(pred_score, self.image_min, self.image_max, thresholds_for_image)
            anomaly_map = self._normalize_with_batch_thresholds(predictions.anomaly_map, self.pixel_min, self.pixel_max, thresholds_raw)
        else:
            pred_score = predictions.pred_score
            anomaly_map = predictions.anomaly_map

        if self.enable_thresholding:
            pred_label = self._apply_threshold(pred_score, self.normalized_image_threshold)
            pred_mask = self._apply_threshold(anomaly_map, self.normalized_pixel_threshold)
        else:
            pred_label = None
            pred_mask = None

        return InferenceBatch(
            pred_label=pred_label,
            pred_score=pred_score,
            pred_mask=pred_mask,
            anomaly_map=anomaly_map,
        )
