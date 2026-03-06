# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomaly map and score generation for L2BT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2BTAnomalyMapGenerator(nn.Module):
    """Generate anomaly maps and image-level scores from L2BT features."""

    def __init__(
        self,
        patch_size: int,
        blur_w_l: int = 5,
        blur_w_u: int = 7,
        blur_pad_l: int = 2,
        blur_pad_u: int = 3,
        blur_repeats_l: int = 5,
        blur_repeats_u: int = 3,
        topk_ratio: float = 0.001,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)

        self.blur_pad_l = blur_pad_l
        self.blur_pad_u = blur_pad_u
        self.blur_repeats_l = blur_repeats_l
        self.blur_repeats_u = blur_repeats_u
        self.topk_ratio = float(topk_ratio)

        weight_l = torch.ones(1, 1, blur_w_l, blur_w_l) / (blur_w_l**2)
        weight_u = torch.ones(1, 1, blur_w_u, blur_w_u) / (blur_w_u**2)
        self.register_buffer("weight_l", weight_l)
        self.register_buffer("weight_u", weight_u)

    def _blur(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        for _ in range(self.blur_repeats_l):
            anomaly_map = F.conv2d(anomaly_map, padding=self.blur_pad_l, weight=self.weight_l)
        for _ in range(self.blur_repeats_u):
            anomaly_map = F.conv2d(anomaly_map, padding=self.blur_pad_u, weight=self.weight_u)
        return anomaly_map

    def _score_topk_mean(self, anomaly_map: torch.Tensor) -> torch.Tensor:
        b, _, h, w = anomaly_map.shape
        n = h * w
        k = max(1, int(n * self.topk_ratio))
        flat = anomaly_map.view(b, -1)  # (B, H*W) because channel=1
        topk_vals = torch.topk(flat, k=k, dim=1).values
        return topk_vals.mean(dim=1)  # (B,)

    def forward(
        self,
        middle_patch: torch.Tensor,
        last_patch: torch.Tensor,
        predicted_middle_patch: torch.Tensor,
        predicted_last_patch: torch.Tensor,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (anomaly_map, pred_score).

        Args:
            middle_patch, last_patch: teacher features
            predicted_middle_patch, predicted_last_patch: student predictions
            output_size: (H, W) of the input image

        Returns:
            anomaly_map: (B, 1, H, W)
            pred_score: (B,)
        """
        b = middle_patch.shape[0]
        h, w = output_size
        h_p, w_p = h // self.patch_size, w // self.patch_size

        middle_anom = (
            (F.normalize(predicted_middle_patch, dim=-1) - F.normalize(middle_patch, dim=-1))
            .pow(2)
            .sum(-1)
            .sqrt()
        )
        last_anom = (
            (F.normalize(predicted_last_patch, dim=-1) - F.normalize(last_patch, dim=-1))
            .pow(2)
            .sum(-1)
            .sqrt()
        )

        combined = middle_anom * last_anom
        if combined.numel() != b * h_p * w_p:
            raise RuntimeError(
                "Patch grid reshape mismatch. "
                f"combined.numel()={combined.numel()}, expected {b*h_p*w_p} "
                f"from output_size={(h, w)} and patch_size={self.patch_size}."
            )

        anomaly_map = combined.view(b, 1, h_p, w_p)
        anomaly_map = F.interpolate(anomaly_map, size=(h, w), mode="bilinear", align_corners=False)
        anomaly_map = self._blur(anomaly_map)

        pred_score = self._score_topk_mean(anomaly_map)
        return anomaly_map, pred_score