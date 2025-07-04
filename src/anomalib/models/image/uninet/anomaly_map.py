"""Utilities for computing anomaly maps."""

# Original Code
# Copyright (c) 2025 Shun Wei
# https://github.com/pangdatangtt/UniNet
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import einops
import numpy as np
import torch
from kornia.filters import gaussian_blur2d
from torch.nn import functional


def weighted_decision_mechanism(
    batch_size: int,
    output_list: list[list[torch.Tensor]],
    alpha: float,
    beta: float,
    out_size: int = 25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute anomaly maps using weighted decision mechanism.

    Args:
        batch_size (int): Batch size.
        output_list (list[list[torch.Tensor]]): List of output tensors.
        alpha (float): Alpha parameter. Used for controlling the upper limit
        beta (float): Beta parameter. Used for controlling the lower limit
        out_size (int): Output size.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Anomaly score and anomaly map.
    """
    total_weights_list = []
    for i in range(batch_size):
        low_similarity_list = [torch.max(output_list[j][i]) for j in range(len(output_list))]
        probs = functional.softmax(torch.tensor(low_similarity_list), dim=0)
        weight_list = []  # set P consists of L high probability values, where L ranges from n-1 to n+1
        for idx, prob in enumerate(probs):
            weight_list.append(low_similarity_list[idx].cpu().numpy()) if prob > torch.mean(probs) else None
        # TODO(ashwinvaidya17): see if all the numpy operations can be replaced with torch operations
        weight = np.max([np.mean(weight_list) * alpha, beta])
        total_weights_list.append(weight)

    assert len(total_weights_list) == batch_size, "The number of weights do not match the number of samples"

    anomaly_map_lists: list[list[torch.Tensor]] = [[] for _ in output_list]
    for idx, output in enumerate(output_list):
        anomaly_map_ = torch.unsqueeze(output, dim=1)  # Bx1xhxw
        # Bx256x256
        anomaly_map_lists[idx] = functional.interpolate(anomaly_map_, out_size, mode="bilinear", align_corners=True)[
            :,
            0,
            :,
            :,
        ]

    anomaly_map: torch.Tensor = sum(anomaly_map_lists)

    anomaly_score = []
    for idx in range(batch_size):
        top_k = int(out_size * out_size * total_weights_list[idx])
        assert top_k >= 1 / (out_size * out_size), "weight can not be smaller than 1 / (H * W)!"

        single_anomaly_score_exp = anomaly_map[idx]
        single_anomaly_score_exp = gaussian_blur2d(
            einops.rearrange(single_anomaly_score_exp, "h w -> 1 1 h w"),  # kornia expects 4D tensor
            kernel_size=(5, 5),
            sigma=(4, 4),
        ).squeeze()
        assert single_anomaly_score_exp.reshape(1, -1).shape[-1] == out_size * out_size, (
            "something wrong with the last dimension of reshaped map!"
        )
        single_map = single_anomaly_score_exp.reshape(1, -1)
        single_anomaly_score = single_map.topk(top_k).values[0][0]
        anomaly_score.append(single_anomaly_score.detach())
    return torch.vstack(anomaly_score), anomaly_map.detach()
