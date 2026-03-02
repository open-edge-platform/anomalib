# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import base64
from collections.abc import Callable

import cv2
import numpy as np
from loguru import logger

from pydantic_models import PredictionLabel, PredictionResponse


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
                    logger.debug(f"Overlay step failed: {e}")

            return visualization
        except Exception as e:
            logger.debug(f"Failed to create visualization: {e}")
            return original_image

    @staticmethod
    def overlay_anomaly_heatmap(
        base_image: np.ndarray,
        prediction: PredictionResponse,
        alpha: float = 0.5,  # TODO: make it configurable by the user
    ) -> np.ndarray:
        """Overlay the anomaly heatmap onto the image using alpha compositing."""
        try:
            anomaly_bytes = base64.b64decode(prediction.anomaly_map)
            overlay = cv2.imdecode(np.frombuffer(anomaly_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if overlay is None or overlay.shape[-1] != 4:
                return base_image

            h, w = base_image.shape[:2]
            if overlay.shape[:2] != (h, w):
                overlay = cv2.resize(overlay, (w, h))

            # Extract RGB (converted from BGR) and alpha channels
            overlay_rgb = cv2.cvtColor(overlay[:, :, :3], cv2.COLOR_BGR2RGB)
            overlay_alpha = (overlay[:, :, 3:4] / 255.0) * alpha

            # Alpha composite
            return (overlay_rgb * overlay_alpha + base_image * (1 - overlay_alpha)).astype(np.uint8)
        except Exception as e:
            logger.debug(f"Failed to overlay heatmap: {e}")
            return base_image

    @staticmethod
    def draw_prediction_label(
        base_image: np.ndarray,
        prediction: PredictionResponse,
        *,
        position: tuple[int, int] = (5, 5),
        font_scale: float = 1.0,
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw the prediction label with a background rectangle for readability."""
        alpha = 0.85
        text_color = (36, 37, 40)
        green = (139, 174, 70)
        red = (255, 86, 98)
        background_color: tuple[int, int, int] = green if prediction.label == PredictionLabel.NORMAL else red
        try:
            label_text = f"{prediction.label.value} {int(prediction.score * 100)}%"
            result = base_image.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            x, y = position[0], position[1] + text_h
            # Create overlay for transparent background
            overlay = result.copy()
            cv2.rectangle(overlay, (x - 8, y - text_h - 8), (x - 8 + text_w + 16, y + 8), background_color, -1)
            result = cv2.addWeighted(result, 1.0 - alpha, overlay, alpha, 0)

            cv2.putText(result, label_text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
            return result
        except Exception as e:
            logger.debug(f"Failed to draw label: {e}")
            return base_image
