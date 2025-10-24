# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""MEBin (Main Element Binarization) adaptive thresholding for anomaly detection.

This module provides the ``MEBin`` class which implements the Main Element
Binarization algorithm designed to address the non-prominence of anomalies
in anomaly maps. MEBin obtains anomaly-centered images by analyzing the
stability of connected components across multiple threshold levels.

The algorithm is particularly effective for:
- Industrial anomaly detection scenarios
- Multi-class anomaly classification tasks
- Cases where anomalies are non-prominent in anomaly maps
- Avoiding the impact of incorrect detections

The threshold is computed by:
1. Adaptively determining threshold search range from anomaly map statistics
2. Sampling anomaly maps at configurable rates across threshold range
3. Counting connected components at each threshold level
4. Finding stable intervals where component count remains constant
5. Selecting threshold from the longest stable interval

MEBin was introduced in "AnomalyNCD: Towards Novel Anomaly Class Discovery
in Industrial Scenarios" (https://arxiv.org/abs/2410.14379).

Example:
    >>> import numpy as np
    >>> from anomalib.metrics.threshold import MEBin
    >>>
    >>> # Create sample anomaly maps with simulated anomalous regions
    >>> anomaly_maps = []
    >>> for i in range(5):
    ...     amap = np.random.rand(128, 128) * 50  # Background noise
    ...     amap[40:80, 40:80] = np.random.rand(40, 40) * 200 + 55  # Anomalous region
    ...     anomaly_maps.append(amap)
    >>>
    >>> # Initialize MEBin with appropriate parameters
    >>> mebin = MEBin(anomaly_maps, sample_rate=8, min_interval_len=3)
    >>>
    >>> # Compute binary masks and thresholds
    >>> binarized_maps, thresholds = mebin.binarize_anomaly_maps()
    >>> print(f"Processed {len(binarized_maps)} maps, thresholds: {thresholds}")
    Processed 5 maps, thresholds: [...]

Note:
    MEBin is designed for industrial scenarios where anomalies may be
    non-prominent. The min_interval_len parameter should be tuned based
    on the expected stability of connected component counts.
"""

from __future__ import annotations

import cv2
import numpy as np
from tqdm import tqdm


class MEBin:
    """MEBin (Main Element Binarization) adaptive thresholding algorithm.

    This class implements the Main Element Binarization algorithm designed
    to address non-prominent anomalies in industrial anomaly detection scenarios.
    MEBin determines optimal thresholds by analyzing the stability of connected
    component counts across different threshold levels to obtain anomaly-centered
    binary representations.

    The algorithm works by:
        - Adaptively determining threshold search ranges from anomaly statistics
        - Sampling anomaly maps at configurable rates across threshold range
        - Counting connected components at each threshold level
        - Identifying stable intervals where component count remains constant
        - Selecting the optimal threshold from the longest stable interval
        - Optionally applying morphological erosion to reduce noise

    Args:
        anomaly_map_list (list[np.ndarray]): List of anomaly map arrays as numpy arrays.
        sample_rate (int, optional): Sampling rate for threshold search. Higher
            values reduce processing time but may affect accuracy.
            Defaults to 4.
        min_interval_len (int, optional): Minimum length of stable intervals.
            Should be tuned based on the expected stability of anomaly score
            distributions.
            Defaults to 4.
        erode (bool, optional): Whether to apply morphological erosion to
            binarized results to reduce noise.
            Defaults to True.

    Example:
        >>> import numpy as np
        >>> from anomalib.metrics.threshold import MEBin
        >>>
        >>> # Create sample anomaly maps with realistic structure
        >>> anomaly_maps = []
        >>> for i in range(3):
        ...     # Background with low anomaly scores
        ...     amap = np.random.rand(64, 64) * 30
        ...     # Add anomalous regions with higher scores
        ...     amap[20:40, 20:40] = np.random.rand(20, 20) * 150 + 100
        ...     anomaly_maps.append(amap)
        >>>
        >>> # Initialize MEBin with custom parameters
        >>> mebin = MEBin(anomaly_maps, sample_rate=4, min_interval_len=3, erode=True)
        >>>
        >>> # Binarize anomaly maps
        >>> binary_masks, thresholds = mebin.binarize_anomaly_maps()
        >>> print(f"Generated {len(binary_masks)} binary masks")
        Generated 3 binary masks
    """

    def __init__(
        self,
        anomaly_map_list: list[np.ndarray],
        sample_rate: int = 4,
        min_interval_len: int = 4,
        erode: bool = True,
    ) -> None:
        self.anomaly_map_list = anomaly_map_list

        self.sample_rate = sample_rate
        self.min_interval_len = min_interval_len
        self.erode = erode

        # Adaptively determine the threshold search range
        self.max_th, self.min_th = self.get_search_range()

    def get_search_range(self) -> tuple[float, float]:
        """Determine the threshold search range adaptively.

        This method analyzes all anomaly maps to determine the minimum and maximum
        threshold values for the binarization process. The search range is based
        on the actual anomaly score distributions in the input maps.

        Returns:
            tuple[float, float]: A tuple containing:
                - max_th: Maximum threshold for binarization.
                - min_th: Minimum threshold for binarization.
        """
        # Get the anomaly scores of all anomaly maps
        anomaly_score_list = [np.max(x) for x in self.anomaly_map_list]

        # Select the maximum and minimum anomaly scores from images
        max_score, min_score = max(anomaly_score_list), min(anomaly_score_list)
        max_th, min_th = max_score, min_score

        return max_th, min_th

    def get_threshold(
        self,
        anomaly_num_sequence: list[int],
        min_interval_len: int,
    ) -> tuple[int, int]:
        """Find the 'stable interval' in the anomaly region number sequence.

        Stable Interval: A continuous threshold range in which the number of connected components remains constant,
        and the length of the threshold range is greater than or equal to the given length threshold
        (min_interval_len).

        Args:
            anomaly_num_sequence (list): Sequence of connected component counts
                at each threshold level, ordered from high to low threshold.
            min_interval_len (int): Minimum length requirement for stable intervals.
                Longer intervals indicate more robust threshold selection.

        Returns:
            tuple[int, int]: A tuple containing (threshold, est_anomaly_num) where
                threshold is the final threshold for binarization and est_anomaly_num
                is the estimated number of anomalies.
        """
        interval_result = {}
        current_index = 0
        while current_index < len(anomaly_num_sequence):
            start = current_index
            value = anomaly_num_sequence[start]
            end = start
            # Move the 'end' pointer forward until a different connected component number is encountered.
            while (
                end < len(anomaly_num_sequence) - 1 and anomaly_num_sequence[end] == anomaly_num_sequence[end + 1]
            ):
                end += 1
            # If the length of the current stable interval is greater than or equal to the given
            # threshold (min_interval_len), and the value is not zero, record this interval.
            if end - start + 1 >= min_interval_len and value != 0:
                if value not in interval_result:
                    interval_result[value] = [(start, end)]
                else:
                    interval_result[value].append((start, end))
            current_index = end + 1

        # If a 'stable interval' exists, calculate the final threshold based on the longest stable interval.
        # If no stable interval is found, it indicates that no anomaly regions exist, and 255 is returned.

        if interval_result:
            # Iterate through the stable intervals, calculating their lengths and corresponding
            # number of connected component.
            count_result = {}
            for anomaly_num in interval_result:
                count_result[anomaly_num] = max(x[1] - x[0] for x in interval_result[anomaly_num])
            est_anomaly_num = max(count_result, key=lambda k: count_result[k])
            est_anomaly_num_interval_result = interval_result[est_anomaly_num]

            # Find the longest stable interval.
            longest_interval = sorted(est_anomaly_num_interval_result, key=lambda x: x[1] - x[0])[-1]

            # Use the endpoint threshold of the longest stable interval as the final threshold.
            index = longest_interval[1]
            threshold = 255 - index * self.sample_rate
            threshold = int(threshold * (self.max_th - self.min_th) / 255 + self.min_th)
            return threshold, est_anomaly_num
        return 255, 0

    def bin_and_erode(self, anomaly_map: np.ndarray, threshold: int) -> np.ndarray:
        """Binarize anomaly map and optionally apply erosion.

        This method converts a continuous anomaly map to a binary mask using
        the specified threshold, and optionally applies morphological erosion
        to reduce noise and smooth the boundaries of anomaly regions.

        The binarization process:
            1. Pixels above threshold become 255 (anomalous)
            2. Pixels below threshold become 0 (normal)
            3. Optional erosion with 6x6 kernel to reduce noise

        Args:
            anomaly_map (numpy.ndarray): Input anomaly map with continuous
                anomaly scores to be binarized.
            threshold (int): Threshold value for binarization. Pixels with
                values above this threshold are considered anomalous.

        Returns:
            numpy.ndarray: Binary mask where 255 indicates anomalous regions
                and 0 indicates normal regions. The result is of type uint8.

        Note:
            Erosion is applied with a 6x6 kernel and 1 iteration to balance
            noise reduction with preservation of anomaly boundaries.
        """
        bin_result = np.where(anomaly_map > threshold, 255, 0).astype(np.uint8)

        # Apply erosion operation to the binarized result
        if self.erode:
            kernel_size = 6
            iter_num = 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            bin_result = cv2.erode(bin_result, kernel, iterations=iter_num)
        return bin_result

    def binarize_anomaly_maps(self) -> tuple[list[np.ndarray], list[int]]:
        """Perform binarization within the given threshold search range.

        Count the number of connected components in the binarized results.
        Adaptively determine the threshold according to the count,
        and perform binarization on the anomaly maps.

        Returns:
            tuple[list[np.ndarray], list[int]]: A tuple containing (binarized_maps, thresholds)
                where binarized_maps is a list of binarized images and thresholds is a list
                of thresholds for each image.
        """
        self.binarized_maps = []
        self.thresholds = []

        for anomaly_map in tqdm(self.anomaly_map_list):
            # Normalize the anomaly map within the given threshold search range.
            if self.max_th == self.min_th:
                # Rare case where all anomaly maps have identical max values
                anomaly_map_norm = np.where(anomaly_map < self.min_th, 0, 255)
            else:
                anomaly_map_norm = np.where(
                    anomaly_map < self.min_th,
                    0,
                    ((anomaly_map - self.min_th) / (self.max_th - self.min_th)) * 255,
                )
            anomaly_num_sequence = []

            # Search for the threshold from high to low within the given range using the specified sampling rate.
            for score in range(255, 0, -self.sample_rate):
                bin_result = self.bin_and_erode(anomaly_map_norm, score)
                num_labels, *_rest = cv2.connectedComponentsWithStats(bin_result, connectivity=8)
                anomaly_num = num_labels - 1
                anomaly_num_sequence.append(anomaly_num)

            # Adaptively determine the threshold based on the anomaly connected component count sequence.
            threshold, _est_anomaly_num = self.get_threshold(anomaly_num_sequence, self.min_interval_len)

            # Binarize the anomaly image based on the determined threshold.
            bin_result = self.bin_and_erode(anomaly_map, threshold)
            self.binarized_maps.append(bin_result)
            self.thresholds.append(threshold)

        return self.binarized_maps, self.thresholds
