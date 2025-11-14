"""MEBin adaptive thresholding algorithm for anomaly detection.

This module provides the ``MEBin`` class which automatically finds the optimal
threshold value by analyzing the stability of connected components across
multiple threshold levels.

The threshold is computed by:
1. Sampling anomaly maps at configurable rates across threshold range
2. Counting connected components at each threshold level
3. Finding stable intervals where component count remains constant
4. Selecting threshold from the longest stable interval

Example:
    >>> from anomalib.metrics.threshold import MEBin
    >>> import numpy as np
    >>> # Create sample anomaly maps
    >>> anomaly_maps = [np.random.rand(256, 256) * 255 for _ in range(10)]
    >>> # Initialize and compute thresholds
    >>> mebin = MEBin(anomaly_map_list=anomaly_maps, sample_rate=4)
    >>> binarized_maps, thresholds = mebin.binarize_anomaly_maps()
    >>> print(f"Computed {len(thresholds)} thresholds")

Note:
    The algorithm works best when anomaly maps contain clear separation between
    normal and anomalous regions. The min_interval_len parameter should be tuned
    based on the expected stability of anomaly score distributions.
"""

import cv2
import numpy as np
from tqdm import tqdm



class MEBin:
    """MEBin adaptive thresholding algorithm for anomaly detection.

    This class implements the MEBin (Minimum Entropy Binarization) algorithm
    which automatically determines optimal thresholds for converting continuous
    anomaly maps to binary masks by analyzing the stability of connected
    component counts across different threshold levels.

    The algorithm works by:
        - Sampling anomaly maps at configurable rates across threshold range
        - Counting connected components at each threshold level
        - Identifying stable intervals where component count remains constant
        - Selecting the optimal threshold from the longest stable interval
        - Optionally applying morphological erosion to reduce noise

    Args:
        anomaly_map_path_list (list, optional): List of file paths to anomaly maps.
            If provided, maps will be loaded as grayscale images. 
            Defaults to None.
        anomaly_map_list (list, optional): List of anomaly map arrays. If provided,
            maps should be numpy arrays. 
            Defaults to None.
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
        >>> from anomalib.metrics.threshold import MEBin
        >>> import numpy as np
        >>> # Create sample anomaly maps
        >>> anomaly_maps = [np.random.rand(256, 256) * 255 for _ in range(10)]
        >>> # Initialize MEBin
        >>> mebin = MEBin(anomaly_map_list=anomaly_maps, sample_rate=4)
        >>> # Compute binary masks and thresholds
        >>> binarized_maps, thresholds = mebin.binarize_anomaly_maps()
    """

    def __init__(self, anomaly_map_path_list=None, sample_rate=4, min_interval_len=4, erode=True):

        self.anomaly_map_path_list = anomaly_map_path_list
        # Load anomaly maps as grayscale images if paths are provided
        self.anomaly_map_list = [cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in self.anomaly_map_path_list]
        
        self.sample_rate = sample_rate
        self.min_interval_len = min_interval_len
        self.erode = erode
        
        # Adaptively determine the threshold search range
        self.max_th, self.min_th = self.get_search_range()
            
        
    def get_search_range(self):
        """Determine the threshold search range adaptively.

        This method analyzes all anomaly maps to determine the minimum and maximum
        threshold values for the binarization process. The search range is based
        on the actual anomaly score distributions in the input maps.

        Returns:
            max_th (int): Maximum threshold for binarization.
            min_th (int): Minimum threshold for binarization.
        """
        # Get the anomaly scores of all anomaly maps
        anomaly_score_list = [np.max(x) for x in self.anomaly_map_list]

        # Select the maximum and minimum anomaly scores from images
        max_score, min_score = max(anomaly_score_list), min(anomaly_score_list)
        max_th, min_th = max_score, min_score

        print(f"Value range: {min_score} - {max_score}")

        return max_th, min_th



    def get_threshold(self, anomaly_num_sequence, min_interval_len):
        """
        Find the 'stable interval' in the anomaly region number sequence.
        Stable Interval: A continuous threshold range in which the number of connected components remains constant, 
        and the length of the threshold range is greater than or equal to the given length threshold (min_interval_len).
        
        Args:
            anomaly_num_sequence (list): Sequence of connected component counts
                at each threshold level, ordered from high to low threshold.
            min_interval_len (int): Minimum length requirement for stable intervals.
                Longer intervals indicate more robust threshold selection.

        Returns:
            threshold (int): The final threshold for binarization.
            est_anomaly_num (int): The estimated number of anomalies.
        """
        interval_result = {}
        current_index = 0
        while current_index < len(anomaly_num_sequence):
            end = current_index 

            start = end 

            # Find the interval where the connected component count remains constant.
            if len(set(anomaly_num_sequence[start:end+1])) == 1 and anomaly_num_sequence[start] != 0:
                # Move the 'end' pointer forward until a different connected component number is encountered.
                while end < len(anomaly_num_sequence)-1 and anomaly_num_sequence[end] == anomaly_num_sequence[end+1]:
                    end += 1
                    current_index += 1
                # If the length of the current stable interval is greater than or equal to the given threshold (min_interval_len), record this interval.
                if end - start + 1 >= min_interval_len:
                    if anomaly_num_sequence[start] not in interval_result:
                        interval_result[anomaly_num_sequence[start]] = [(start, end)]
                    else:
                        interval_result[anomaly_num_sequence[start]].append((start, end))
            current_index += 1

        """
        If a 'stable interval' exists, calculate the final threshold based on the longest stable interval.
        If no stable interval is found, it indicates that no anomaly regions exist, and 255 is returned.
        """

        if interval_result:
            # Iterate through the stable intervals, calculating their lengths and corresponding number of connected component.
            count_result = {}
            for anomaly_num in interval_result:
                count_result[anomaly_num] = max([x[1] - x[0] for x in interval_result[anomaly_num]])
            est_anomaly_num = max(count_result, key=count_result.get)
            est_anomaly_num_interval_result = interval_result[est_anomaly_num]

            # Find the longest stable interval.
            longest_interval = sorted(est_anomaly_num_interval_result, key=lambda x: x[1] - x[0])[-1]

            # Use the endpoint threshold of the longest stable interval as the final threshold.
            index = longest_interval[1]
            threshold = 255 - index * self.sample_rate
            threshold = int(threshold*(self.max_th - self.min_th)/255 + self.min_th)
            return threshold, est_anomaly_num
        else:
            return 255, 0
        
        
    def bin_and_erode(self, anomaly_map, threshold):
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
    

    def binarize_anomaly_maps(self):
        """
        Perform binarization within the given threshold search range,
        count the number of connected components in the binarized results.
        Adaptively determine the threshold according to the count,
        and perform binarization on the anomaly maps.
        
        Returns:
            binarized_maps (list): List of binarized images.
            thresholds (list): List of thresholds for each image.
        """
        self.binarized_maps = []
        self.thresholds = []
        
        for i, anomaly_map in enumerate(tqdm(self.anomaly_map_list)):
            # Normalize the anomaly map within the given threshold search range.
            anomaly_map_norm = np.where(anomaly_map < self.min_th, 0, ((anomaly_map - self.min_th) / (self.max_th - self.min_th)) * 255)
            anomaly_num_sequence = []

            # Search for the threshold from high to low within the given range using the specified sampling rate.
            for score in range(255, 0, -self.sample_rate):
                bin_result = self.bin_and_erode(anomaly_map_norm, score)
                num_labels, *rest = cv2.connectedComponentsWithStats(bin_result, connectivity=8)
                anomaly_num = num_labels - 1
                anomaly_num_sequence.append(anomaly_num)

            # Adaptively determine the threshold based on the anomaly connected component count sequence.
            threshold, est_anomaly_num = self.get_threshold(anomaly_num_sequence, self.min_interval_len)

            # Binarize the anomaly image based on the determined threshold.
            bin_result = self.bin_and_erode(anomaly_map, threshold)
            self.binarized_maps.append(bin_result)
            self.thresholds.append(threshold)

        return self.binarized_maps, self.thresholds