# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime

import cv2
import numpy as np
from anomalib.data import NumpyImageBatch as PredictionResult
from loguru import logger

from pydantic_models.sink import FolderSinkConfig, OutputFormat
from services.dispatchers.base import BaseDispatcher


class FolderDispatcher(BaseDispatcher):
    """FolderDispatcher allows outputting to a folder in the local filesystem."""

    def __init__(self, output_config: FolderSinkConfig) -> None:
        """
        Initialize the FolderDispatcher.
        Args:
            output_config: Configuration for the output destination
        """
        super().__init__(output_config=output_config)
        self.output_folder = output_config.folder_path
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def _write_image_to_file(image: np.ndarray, file_path: str) -> None:
        with open(file_path, "wb") as f:
            success, img_buf = cv2.imencode(".png", image)
            if success:
                f.write(img_buf.tobytes())
            else:
                logger.error(f"Failed to encode image for {file_path}")

    @staticmethod
    def _write_predictions_to_file(predictions: str, file_path: str) -> None:
        with open(file_path, "w") as f:
            f.write(predictions)

    def _dispatch(
        self,
        original_image: np.ndarray,
        image_with_visualization: np.ndarray,
        predictions: PredictionResult,
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # up to milliseconds
        image_orig_file = os.path.join(self.output_folder, f"{timestamp}-original.jpg")
        image_viz_file = os.path.join(self.output_folder, f"{timestamp}-pred.jpg")
        pred_txt_file = os.path.join(self.output_folder, f"{timestamp}-pred.txt")

        logger.trace(f"Saving results to folder for timestamp '{timestamp}' to folder '{self.output_folder}'")

        if OutputFormat.IMAGE_ORIGINAL in self.output_formats:
            self._write_image_to_file(original_image, image_orig_file)
        if OutputFormat.IMAGE_WITH_PREDICTIONS in self.output_formats:
            self._write_image_to_file(image_with_visualization, image_viz_file)
        if OutputFormat.PREDICTIONS in self.output_formats:
            self._write_predictions_to_file(str(predictions), pred_txt_file)
