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
import torch
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components.filters import GaussianBlur2d


def weighted_decision_mechanism(
    batch_size: int,
    output_list: list[torch.Tensor],
    alpha: float,
    beta: float,
    output_size: tuple[int, int] = (256, 256),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute anomaly maps using weighted decision mechanism.

    Args:
        batch_size (int): Batch size.
        output_list (list[torch.Tensor]): List of output tensors, each with shape [batch_size, H, W].
        alpha (float): Alpha parameter. Used for controlling the upper limit
        beta (float): Beta parameter. Used for controlling the lower limit
        output_size (tuple[int, int]): Output size.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Anomaly score and anomaly map.
    """
    # Convert to tensor operations to avoid SequenceConstruct/ConcatFromSequence
    device = output_list[0].device
    num_outputs = len(output_list)

    # Pre-allocate tensors instead of using lists
    total_weights = torch.zeros(batch_size, device=device)
    gaussian_blur = GaussianBlur2d(sigma=4.0, kernel_size=(5, 5), channels=1).to(device)

    # Process each batch item individually
    for i in range(batch_size):
        # Get max value from each output for this batch item
        # Create tensor directly from max values to avoid list operations
        max_values = torch.zeros(num_outputs, device=device)
        for j, output_tensor in enumerate(output_list):
            max_values[j] = torch.max(output_tensor[i])

        probs = F.softmax(max_values, dim=0)

        # Use tensor operations instead of list filtering
        prob_mean = torch.mean(probs)
        mask = probs > prob_mean

        if mask.any():
            weight_tensor = max_values[mask]
            weight = torch.max(torch.stack([torch.mean(weight_tensor) * alpha, torch.tensor(beta, device=device)]))
        else:
            weight = torch.tensor(beta, device=device)

        total_weights[i] = weight

    # Process anomaly maps using tensor operations
    # Pre-allocate the processed anomaly maps tensor
    processed_anomaly_maps = torch.zeros(batch_size, *output_size, device=device)

    # Process each output tensor separately due to different spatial dimensions
    for output_tensor in output_list:
        # Interpolate current output to target size
        # Add channel dimension for interpolation: [batch_size, H, W] -> [batch_size, 1, H, W]
        output_resized = F.interpolate(
            output_tensor.unsqueeze(1),
            output_size,
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)  # [batch_size, H_out, W_out]

        # Add to accumulated anomaly maps
        processed_anomaly_maps += output_resized

    # Pre-allocate anomaly scores tensor instead of using list
    anomaly_scores = torch.zeros(batch_size, device=device)

    for idx in range(batch_size):
        top_k = int(output_size[0] * output_size[1] * total_weights[idx])
        top_k = max(top_k, 1)  # Ensure at least 1 element

        single_anomaly_score_exp = processed_anomaly_maps[idx]
        single_anomaly_score_exp = gaussian_blur(einops.rearrange(single_anomaly_score_exp, "h w -> 1 1 h w"))
        single_anomaly_score_exp = single_anomaly_score_exp.squeeze()

        # Flatten and get top-k values
        single_map_flat = single_anomaly_score_exp.view(-1)
        top_k_values = torch.topk(single_map_flat, top_k).values
        single_anomaly_score = top_k_values[0] if len(top_k_values) > 0 else torch.tensor(0.0, device=device)
        anomaly_scores[idx] = single_anomaly_score.detach()

    return anomaly_scores.unsqueeze(1), processed_anomaly_maps.detach()
