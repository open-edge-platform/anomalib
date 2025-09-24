# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import base64
import logging
from collections.abc import Callable

import cv2
import numpy as np

from pydantic_models import PredictionResponse

logger = logging.getLogger(__name__)


class Visualizer:
    @classmethod
    def overlay_predictions(
        cls,
        original_image: np.ndarray,
        prediction: PredictionResponse,
        *overlays: Callable[..., np.ndarray],
        **kwargs: object,
    ) -> np.ndarray:
        """Create a visualization by applying a sequence of overlay functions.

        Each overlay is a callable that accepts (image, prediction) and returns a new image.
        If no overlays are provided, the default is anomaly heatmap followed by label.
        """
        try:
            visualization = original_image.copy()

            # Default overlays if none provided
            if not overlays:
                overlays = (cls.overlay_anomaly_heatmap, cls.draw_prediction_label)

            for overlay in overlays:
                try:
                    visualization = overlay(visualization, prediction, **kwargs)
                except Exception as e:  # continue other overlays even if one fails
                    logger.debug("Overlay step failed: %s", e)

            return visualization
        except Exception as e:
            logger.debug("Failed to create visualization: %s", e)
            return original_image

    @staticmethod
    def overlay_anomaly_heatmap(
        base_image: np.ndarray,
        prediction: PredictionResponse,
        threshold_value: int = 128,
        alpha: float = 0.25,
    ) -> np.ndarray:
        """Overlay the anomaly heatmap onto the image.

        Steps:
        - Decode base64 anomaly map to an image
        - Convert to grayscale if needed, ensure uint8
        - Apply JET colormap and threshold mask
        - Blend onto the base image using alpha
        """
        try:
            anomaly_map_base64 = prediction.anomaly_map
            result = base_image.copy()
            try:
                anomaly_png_bytes = base64.b64decode(anomaly_map_base64)
                anomaly_np = np.frombuffer(anomaly_png_bytes, dtype=np.uint8)
                anomaly_img = cv2.imdecode(anomaly_np, cv2.IMREAD_UNCHANGED)
            except Exception:
                return result

            if anomaly_img is None:
                return result

            try:
                if anomaly_img.ndim == 3 and anomaly_img.shape[2] > 1:
                    anomaly_gray = cv2.cvtColor(anomaly_img, cv2.COLOR_BGR2GRAY)
                else:
                    anomaly_gray = anomaly_img

                if anomaly_gray.dtype != np.uint8:
                    anomaly_gray = anomaly_gray.astype(np.uint8)

                heatmap = cv2.applyColorMap(anomaly_gray, cv2.COLORMAP_JET)
                heatmap_resized = cv2.resize(heatmap, (result.shape[1], result.shape[0]))

                mask_gray = cv2.resize(anomaly_gray, (result.shape[1], result.shape[0]))
                mask_bool = mask_gray >= threshold_value

                masked_heatmap = np.zeros_like(heatmap_resized)
                try:
                    masked_heatmap[mask_bool] = heatmap_resized[mask_bool]
                except Exception as e:
                    logger.debug("Failed to apply heatmap mask: %s", e)

                result = cv2.addWeighted(result, 1.0, masked_heatmap, alpha, 0)
            except Exception as e:
                logger.debug("Failed to overlay heatmap: %s", e)
            return result
        except Exception as e:
            logger.debug("Failed in overlay_anomaly_heatmap: %s", e)
            return base_image

    @staticmethod
    def draw_prediction_label(
        base_image: np.ndarray,
        prediction: PredictionResponse,
        *,
        position: tuple[int, int] = (10, 20),
        font_scale: float = 2.0,
        thickness: int = 3,
        text_color: tuple[int, int, int] = (0, 255, 0),
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """Draw the prediction label with a background rectangle for readability."""
        try:
            label_text = f"{prediction.label.value} ({prediction.score:.3f})"
            result = base_image.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            x, y = position[0], position[1] + text_h
            cv2.rectangle(result, (x - 8, y - text_h - 8), (x - 8 + text_w + 16, y + 8), background_color, -1)
            cv2.putText(result, label_text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
            return result
        except Exception as e:
            logger.debug("Failed to draw label: %s", e)
            return base_image
