# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np

logger = logging.getLogger(__name__)


class Visualizer:
    @staticmethod
    def overlay_predictions(original_image: np.ndarray) -> np.ndarray:
        """Overlay predictions on the original image based on the type of predictions."""
        try:
            # TODO: implement for video stream predictions
            visualization = original_image
            if visualization is None:
                # If no visualization could be created, return the original image
                return original_image
        except Exception as e:
            logger.exception("An error occurred while creating visualization, returning original image.", exc_info=e)
            return original_image
        return visualization
