# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MEBin (Main Element Binarization) algorithm for anomaly map thresholding.

MEBin is an adaptive binarization method introduced in AnomalyNCD that determines
per-image thresholds by finding "stable intervals" in the connected-component
count as the threshold is swept from high to low.

The search range is determined by per-image peak anomaly scores:   s_max
is the maximum peak across images and   s_min   is the minimum peak.  Values
below   s_min   are clipped to 0, and the range   [s_min, s_max]   is mapped
to   [0, 255]  .  This ensures the threshold sweep focuses on the region where
anomalies actually appear, matching the reference implementation.

Reference:
    "AnomalyNCD: Towards Novel Anomaly Class Discovery in
    Industrial Scenarios", CVPR 2025.
    https://arxiv.org/abs/2410.14379
    https://github.com/HUST-SLOW/AnomalyNCD

The implementation uses kornia (instead of CV2) for connected-component analysis and morphological
erosion to be compatible with ONNX/OpenVINO export.

Example:
    >>> import torch
    >>> from anomalib.post_processing.mebin import mebin_binarize
    >>> anomaly_maps = torch.rand(4, 1, 256, 256)
    >>> masks, thresholds = mebin_binarize(anomaly_maps)
"""

import torch
from kornia.contrib import connected_components
from kornia.morphology import erosion


def _count_connected_components(
    binary_mask: torch.Tensor,
    num_iterations: int = 1000,
) -> torch.Tensor:
    """Count the number of foreground connected components per image.

    Args:
        binary_mask: Binary mask tensor of shape   (B, 1, H, W)   with values
            in   {0, 1}   (float).
        num_iterations: Number of iterations for kornia's connected component
            algorithm. Defaults to 1000.

    Returns:
        torch.Tensor: Integer tensor of shape (B,) with the number of
            foreground connected components in each image.
    """
    labels = connected_components(binary_mask.float(), num_iterations=num_iterations)
    batch_size = labels.shape[0]
    counts = torch.zeros(batch_size, device=labels.device, dtype=torch.long)
    for i in range(batch_size):
        # Unique labels excluding background (0)
        unique_labels = labels[i].unique()
        counts[i] = (unique_labels > 0).sum()
    return counts


def _erode(
    binary_mask: torch.Tensor,
    kernel_size: int = 6,
) -> torch.Tensor:
    """Apply morphological erosion to a binary mask using kornia.

    Args:
        binary_mask: Binary mask tensor of shape   (B, 1, H, W)   with values
            in   {0, 1}   (float).
        kernel_size: Size of the square erosion kernel. Defaults to   6  .

    Returns:
        torch.Tensor: Eroded binary mask of the same shape.
    """
    kernel = torch.ones(kernel_size, kernel_size, device=binary_mask.device)
    return erosion(binary_mask, kernel)


def _find_stable_threshold(
    component_counts: list[int],
    min_interval_len: int,
    sample_rate: int,
) -> tuple[int, int]:
    """Find the binarization threshold from the connected-component count sequence.

    A stable interval is a contiguous range of threshold indices where the
    connected-component count is constant (and non-zero), with length ≥
    `min_interval_len`. The threshold at the endpoint of the longest stable
    interval is selected.

    Args:
        component_counts: List of foreground connected-component counts, ordered
            from the highest threshold to the lowest.
        min_interval_len: Minimum length of a stable interval to be considered.
        sample_rate: Step size used during the threshold sweep (needed to map
            the index back to a threshold value in [0, 255]).

    Returns:
        tuple[int, int]:   (threshold, est_anomaly_num)   where   threshold   is
            in the [0, 255] normalised range and   est_anomaly_num   is the
            connected-component count at that threshold.  Returns   (255, 0)
            when no stable interval is found (i.e. no anomaly).
    """
    interval_result: dict[int, list[tuple[int, int]]] = {}
    idx = 0
    n = len(component_counts)

    while idx < n:
        start = idx
        value = component_counts[start]

        if value == 0:
            idx += 1
            continue

        # Extend to the end of the constant run.
        end = start
        while end < n - 1 and component_counts[end + 1] == value:
            end += 1

        length = end - start + 1
        if length >= min_interval_len:
            interval_result.setdefault(value, []).append((start, end))

        # Advance past this run.
        idx = end + 1

    if not interval_result:
        return 255, 0

    # For each component count, find the longest interval.
    best_count: int | None = None
    best_length = -1
    for count, intervals in interval_result.items():
        max_len = max(e - s + 1 for s, e in intervals)
        if max_len > best_length:
            best_length = max_len
            best_count = count

    assert best_count is not None
    # Within the best count, pick the longest interval.
    intervals = interval_result[best_count]
    longest = max(intervals, key=lambda x: x[1] - x[0] + 1)
    # Threshold value from the endpoint index.
    threshold = 255 - longest[1] * sample_rate
    return max(threshold, 0), best_count


def mebin_binarize(
    anomaly_maps: torch.Tensor,
    sample_rate: int = 4,
    min_interval_len: int = 4,
    erode: bool = True,
    kernel_size: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MEBin adaptive binarization to a batch of anomaly maps.

    The algorithm follows the reference implementation in AnomalyNCD:

    1. Compute the search range   [s_min, s_max]   from   per-image peak scores
       (i.e.   s_max = max(per-image max)   and   s_min = min(per-image max)  ).
       This clips background pixels below the noise floor and focuses the
       threshold sweep on the region where anomalies actually appear.
    2. Per-image: clip pixels below   s_min   to 0 and linearly map
         [s_min, s_max]   →   [0, 255]  .
    3. Sweep thresholds from 255 → 0 (step   sample_rate  ). At each threshold,
       binarize → optionally erode → count connected components.
    4. Find the   stable interval   in the component-count sequence.
    5. Select the threshold at the endpoint of the longest stable interval.
    6. Map the threshold back to the   original   score space, binarize, and
       return binary masks and per-image thresholds.

    Args:
        anomaly_maps: Anomaly score maps of shape   (B, 1, H, W)  .
        sample_rate: Step size for the threshold sweep. Defaults to   4  .
        min_interval_len: Minimum length of a stable interval. Defaults to   4  .
        erode: Whether to apply morphological erosion before counting connected
            components. Defaults to True.
        kernel_size: Size of the erosion kernel (only used when erode=True).
            Defaults to 6.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - masks : Binary masks of shape   (B, 1, H, W)   (float, 0/1).
            - thresholds : Per-image thresholds of shape   (B,)   in the
              original anomaly-score space.

    Example:
        >>> import torch
        >>> maps = torch.rand(4, 1, 256, 256)
        >>> masks, thresholds = mebin_binarize(maps)
        >>> masks.shape
        torch.Size([4, 1, 256, 256])
        >>> thresholds.shape
        torch.Size([4])
    """
    if anomaly_maps.dim() != 4 or anomaly_maps.shape[1] != 1:
        msg = f"Expected anomaly_maps of shape (B, 1, H, W), got {anomaly_maps.shape}"
        raise ValueError(msg)

    device = anomaly_maps.device
    batch_size = anomaly_maps.shape[0]

    # --- Step 1: search range from per-image peak anomaly scores ---
    # Following the reference: max_th = max(per-image max), min_th = min(per-image max).
    # This focuses the sweep on the score range where anomalies appear, clipping
    # the background noise floor below min_th.
    per_image_peaks = anomaly_maps.amax(dim=(-2, -1)).squeeze(-1)  # (B,)
    s_max = per_image_peaks.max()
    s_min = per_image_peaks.min()
    score_range = s_max - s_min
    if score_range == 0:
        # All images have the same peak → use global min/max as fallback.
        global_min = anomaly_maps.amin()
        score_range = s_max - global_min
        if score_range == 0:
            # Truly uniform maps → no anomaly anywhere.
            return torch.zeros_like(anomaly_maps), s_max.expand(batch_size)
        s_min = global_min

    # --- Step 2: normalise to [0, 255], clipping below s_min to 0 ---
    maps_norm = torch.where(
        anomaly_maps < s_min,
        torch.zeros_like(anomaly_maps),
        ((anomaly_maps - s_min) / score_range * 255.0).clamp(0, 255),
    )

    # Pre-compute erosion kernel once.
    erosion_kernel = torch.ones(kernel_size, kernel_size, device=device) if erode else None

    masks = torch.zeros_like(anomaly_maps)
    thresholds_raw = torch.full((batch_size,), s_max.item(), device=device, dtype=anomaly_maps.dtype)

    # --- Per-image threshold search ---
    for i in range(batch_size):
        single_map = maps_norm[i : i + 1]  # (1, 1, H, W)
        component_counts: list[int] = []

        for score in range(255, 0, -sample_rate):
            # Binarize at this threshold.
            binary = (single_map > score).float()
            # Optionally erode.
            if erode and erosion_kernel is not None:
                binary = erosion(binary, erosion_kernel)
                binary = (binary > 0.5).float()
            # Count connected components.
            n_components = _count_connected_components(binary)
            component_counts.append(int(n_components.item()))

        # Find the stable-interval threshold (in [0, 255] space).
        thresh_norm, _est_num = _find_stable_threshold(
            component_counts,
            min_interval_len,
            sample_rate,
        )

        # Map threshold back to original score space.
        thresh_raw = thresh_norm / 255.0 * score_range + s_min
        thresholds_raw[i] = thresh_raw

        # Binarize the original (un-normalised) map at the raw threshold.
        masks[i] = (anomaly_maps[i] > thresh_raw).float()

    return masks, thresholds_raw
