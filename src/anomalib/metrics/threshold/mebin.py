import cv2
import numpy as np
from tqdm import tqdm

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False



class MEBin:
    def __init__(self, anomaly_map_list=None, sample_rate=4, min_interval_len=4, erode=True):
        '''
        Get the anomaly maps and the threshold range.
        
        Args:
            anomaly_map_list (torch.Tensor): List of anomaly maps.
            sample_rate (int): Sampling rate for threshold search.
            min_interval_len (int): Minimum length of the stable interval.
            erode (bool): Whether to apply erosion operation to the binarized result.
        '''
        self.anomaly_map_list = anomaly_map_list
        
        self.sample_rate = sample_rate
        self.min_interval_len = min_interval_len
        self.erode = erode
        
        # Adaptively determine the threshold search range
        self.max_th, self.min_th = self.get_search_range()
            
        
    def get_search_range(self):
        '''
        Determine the threshold search range based on the maximum and minimum anomaly scores of the anomaly score images to be binarized, 
        as well as parameters from the configuration file.
        
        Returns:
            max_th (int): Maximum threshold for binarization.
            min_th (int): Minimum threshold for binarization.
        '''
        # Get the anomaly scores of all anomaly maps
        # 支持 tensor 和 numpy 数组
        anomaly_score_list = []
        for x in self.anomaly_map_list:
            if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
                score = x.max().item()
            else:
                score = np.max(x)
            anomaly_score_list.append(score)

        # Select the maximum and minimum anomaly scores from images
        max_score, min_score = max(anomaly_score_list), min(anomaly_score_list)
        max_th, min_th = max_score, min_score

        print(f"Value range: {min_score} - {max_score}")

        return max_th, min_th


    def get_threshold(self, anomaly_num_sequence, min_interval_len):
        '''
        Find the 'stable interval' in the anomaly region number sequence.
        Stable Interval: A continuous threshold range in which the number of connected components remains constant, 
        and the length of the threshold range is greater than or equal to the given length threshold (min_interval_len).
        
        Args:
            anomaly_num_sequence (int): The number of connected components in the binarized map at each threshold.
            min_interval_len (int): The minimum length of the stable interval.
        Returns:
            threshold (int): The final threshold for binarization.
            est_anomaly_num (int): The estimated number of anomalies.
        '''
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

        '''
        If a 'stable interval' exists, calculate the final threshold based on the longest stable interval.
        If no stable interval is found, it indicates that no anomaly regions exist, and 255 is returned.
        '''

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
            # threshold = int(threshold*(self.max_th - self.min_th)/255.0 + self.min_th)
            threshold = int(threshold * (float(self.max_th) - float(self.min_th)) / 255.0 + float(self.min_th))
            return threshold, est_anomaly_num
        else:
            return 255, 0
        
        
    def bin_and_erode(self, anomaly_map, threshold):
        '''
        Binarize the anomaly map based on the given threshold.
        Apply erosion operation to the binarized result to reduce noise, as specified in the configuration file.
        
        Args:
            anomaly_map: The anomaly map to be binarized (numpy array or torch.Tensor).
            threshold: The threshold used for binarization.
        Returns:
            bin_result: The binarized result after applying the erosion operation(optiaonal).
        '''
        # 如果是 tensor，转换为 numpy 数组
        if TORCH_AVAILABLE and isinstance(anomaly_map, torch.Tensor):
            anomaly_map = anomaly_map.cpu().numpy()
        
        bin_result = np.where(anomaly_map > threshold, 255, 0).astype(np.uint8)

        # Apply erosion operation to the binarized result
        if self.erode:
            kernel_size = 6
            iter_num = 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            bin_result = cv2.erode(bin_result, kernel, iterations=iter_num)
        return bin_result
    

    def binarize_anomaly_maps(self):
        '''
        Perform binarization within the given threshold search range,
        count the number of connected components in the binarized results.
        Adaptively determine the threshold according to the count,
        and perform binarization on the anomaly maps.
        
        Returns:
            binarized_maps (list): List of binarized images.
            thresholds (list): List of thresholds for each image.
        '''
        self.binarized_maps = []
        self.thresholds = []
        
        
        for i, anomaly_map in enumerate(tqdm(self.anomaly_map_list)):
            # 如果是 tensor，转换为 numpy 数组进行处理
            if TORCH_AVAILABLE and isinstance(anomaly_map, torch.Tensor):
                anomaly_map_np = anomaly_map.cpu().numpy()
            else:
                anomaly_map_np = anomaly_map
            
            # Normalize the anomaly map within the given threshold search range.
            anomaly_map_norm = np.where(anomaly_map_np < self.min_th, 0, ((anomaly_map_np - self.min_th) / (self.max_th - self.min_th)) * 255)
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
            # 使用原始的 anomaly_map（可能是 tensor 或 numpy）
            bin_result = self.bin_and_erode(anomaly_map, threshold)
            self.binarized_maps.append(bin_result)
            self.thresholds.append(threshold)

        return self.binarized_maps, self.thresholds