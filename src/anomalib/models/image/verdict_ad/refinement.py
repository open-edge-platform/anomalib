# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Detector-independent global and local refinement from VerdictAD.

Adapted from https://github.com/sgstq0811/VerdictAD-Jev-AD (Apache-2.0).
"""

import math

import torch
from torch.nn import functional as F  # noqa: N812

from anomalib.data import InferenceBatch


def refine_predictions(
    predictions: InferenceBatch,
    global_logits: torch.Tensor,
    window_logits: torch.Tensor,
    boxes: list[tuple[float, float, float, float]],
    alpha: float = 0.5,
    gamma: float = 0.25,
) -> InferenceBatch:
    """Refine raw detector outputs using the paper's GVR and LVR equations.

    Args:
        predictions (InferenceBatch): Nonnegative raw scores ``(B,)`` and maps
            ``(B, H, W)`` or ``(B, 1, H, W)`` in the query image's coordinates.
        global_logits (torch.Tensor): Reference-free and conditioned logits, ``(B, 2)``.
            May have shape ``(B, 0)`` when global refinement is disabled.
        window_logits (torch.Tensor): Defect log-odds for each window, ``(B, N)``.
        boxes (list[tuple[float, float, float, float]]): Normalized XYXY windows.
            Crop and boxed-view decisions repeat the same coordinates.
        alpha (float): Global gate exponent. Zero disables GVR.
        gamma (float): Local gate exponent. Zero disables LVR.

    Returns:
        InferenceBatch: Refined scores and maps, with stale labels/masks removed.

    Raises:
        ValueError: If inputs are invalid or windows do not cover the map.
    """
    if any(not math.isfinite(value) or value < 0 for value in (alpha, gamma)):
        msg = "alpha and gamma must be finite and nonnegative."
        raise ValueError(msg)
    scores, maps = predictions.pred_score, predictions.anomaly_map
    if scores is None or maps is None:
        msg = "VerdictAD requires both raw pred_score and anomaly_map."
        raise ValueError(msg)
    if scores.ndim != 1 or maps.ndim not in {3, 4} or maps.shape[0] != scores.shape[0]:
        msg = "Expected scores (B,) and maps (B, H, W) or (B, 1, H, W)."
        raise ValueError(msg)
    if maps.ndim == 4 and maps.shape[1] != 1:
        msg = "Anomaly maps must have one channel."
        raise ValueError(msg)
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in (scores, maps)):
        msg = "Detector outputs must be finite floating-point tensors."
        raise ValueError(msg)
    if (scores < 0).any() or (maps < 0).any():
        msg = "VerdictAD requires nonnegative raw detector outputs."
        raise ValueError(msg)
    batch_size = scores.shape[0]
    global_shapes = {(batch_size, 2)} if alpha else {(batch_size, 0), (batch_size, 2)}
    if global_logits.shape not in global_shapes or window_logits.shape != (batch_size, len(boxes)):
        msg = "Decision logits must have shapes (B, 2) and (B, number of boxes)."
        raise ValueError(msg)
    if not torch.isfinite(global_logits).all() or not torch.isfinite(window_logits).all():
        msg = "Decision logits must be finite."
        raise ValueError(msg)
    refined_scores = scores
    if alpha:
        dtype = torch.promote_types(torch.promote_types(scores.dtype, global_logits.dtype), torch.float32)
        logits = global_logits.to(device=scores.device, dtype=dtype)
        mean_logit = logits[:, 0] * 0.5 + logits[:, 1] * 0.5
        gate = (F.logsigmoid(mean_logit) * alpha).exp()
        refined_scores = (scores * gate).to(scores.dtype)
    refined_maps = maps
    if gamma:
        height, width = maps.shape[-2:]
        dtype = torch.promote_types(torch.promote_types(maps.dtype, window_logits.dtype), torch.float32)
        weights = torch.zeros((batch_size, height, width), device=maps.device, dtype=dtype)
        counts = torch.zeros((height, width), device=maps.device, dtype=dtype)
        logits = window_logits.to(device=maps.device, dtype=dtype)
        probabilities = (F.logsigmoid(logits) * gamma).exp()
        for index, (x0, y0, x1, y1) in enumerate(boxes):
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                msg = "Boxes must have normalized, ordered XYXY coordinates."
                raise ValueError(msg)
            left, top = int(x0 * width), int(y0 * height)
            right, bottom = round(x1 * width), round(y1 * height)
            weights[:, top:bottom, left:right] += probabilities[:, index, None, None]
            counts[top:bottom, left:right] += 1
        if (counts == 0).any():
            msg = "Decision windows must cover every anomaly-map pixel."
            raise ValueError(msg)
        weights /= counts
        if maps.ndim == 4:
            weights = weights.unsqueeze(1)
        refined_maps = (maps * weights).to(maps.dtype)
    return InferenceBatch(pred_score=refined_scores, anomaly_map=refined_maps)
