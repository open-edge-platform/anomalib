# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod

import numpy as np
from model_api.models import AnomalyResult, ClassificationResult, DetectedKeypoints, ImageResultWithSoftPrediction
from model_api.models.result import DetectionResult, InstanceSegmentationResult, Label, Result
from model_api.visualizer import BoundingBox, Flatten, Polygon
from model_api.visualizer.scene import (
    AnomalyScene,
    ClassificationScene,
    DetectionScene,
    InstanceSegmentationScene,
    KeypointScene,
    SegmentationScene,
)
from PIL import Image

from utils.singleton import Singleton

logger = logging.getLogger(__name__)


class VisualizerCreator(ABC):
    """Abstract base class for visualizer creators."""

    @abstractmethod
    def create_visualization(self, original_image: np.ndarray, predictions: Result) -> np.ndarray:
        """Create a visualization of the predictions on the original image."""


class DetectionVisualizerCreator(VisualizerCreator):
    """Creator for detection visualizations."""

    def create_visualization(self, original_image: np.ndarray, predictions: DetectionResult) -> np.ndarray:
        """Create a visualization of the detection predictions on the original image."""
        image_pil = Image.fromarray(original_image)
        detection_scene = DetectionScene(image=image_pil, result=predictions, layout=Flatten(BoundingBox, Label))
        rendered_detections_pil = detection_scene.render()
        return np.array(rendered_detections_pil)


class InstanceSegmentationVisualizerCreator(VisualizerCreator):
    """Creator for instance segmentation visualizations."""

    def create_visualization(self, original_image: np.ndarray, predictions: InstanceSegmentationResult) -> np.ndarray:
        """Create a visualization of the instance segmentation predictions on the original image."""
        image_pil = Image.fromarray(original_image)
        segmentation_scene = InstanceSegmentationScene(
            image=image_pil, result=predictions, layout=Flatten(Polygon, Label)
        )
        rendered_segmentation_pil = segmentation_scene.render()
        return np.array(rendered_segmentation_pil)


class AnomalyDetectionVisualizerCreator(VisualizerCreator):
    """Creator for anomaly detection visualizations."""

    def create_visualization(self, original_image: np.ndarray, predictions: AnomalyResult) -> np.ndarray:
        """Create a visualization of the anomaly detection predictions on the original image."""
        image_pil = Image.fromarray(original_image)
        anomaly_detection_scene = AnomalyScene(image=image_pil, result=predictions, layout=Flatten(BoundingBox, Label))
        rendered_anomaly_detection_pil = anomaly_detection_scene.render()
        return np.array(rendered_anomaly_detection_pil)


class ClassificationVisualizerCreator(VisualizerCreator):
    """Creator for classification visualizations."""

    def create_visualization(self, original_image: np.ndarray, predictions: ClassificationResult) -> np.ndarray:
        """Create a visualization of the classification predictions on the original image."""
        image_pil = Image.fromarray(original_image)
        classification_scene = ClassificationScene(
            image=image_pil,
            result=predictions,
        )
        rendered_classification_pil = classification_scene.render()
        return np.array(rendered_classification_pil)


class SegmentationVisualizerCreator(VisualizerCreator):
    """Creator for segmentation visualizations."""

    def create_visualization(
        self, original_image: np.ndarray, predictions: ImageResultWithSoftPrediction
    ) -> np.ndarray:
        """Create a visualization of the segmentation predictions on the original image."""
        image_pil = Image.fromarray(original_image)
        segmentation_scene = SegmentationScene(image=image_pil, result=predictions, layout=Flatten(Polygon, Label))
        rendered_segmentation_pil = segmentation_scene.render()
        return np.array(rendered_segmentation_pil)


class KeypointVisualizerCreator(VisualizerCreator):
    """Creator for keypoint visualizations."""

    def create_visualization(self, original_image: np.ndarray, predictions: DetectedKeypoints) -> np.ndarray:
        """Create a visualization of the keypoint predictions on the original image."""
        image_pil = Image.fromarray(original_image)
        keypoint_scene = KeypointScene(
            image=image_pil,
            result=predictions,
        )
        rendered_keypoint_pil = keypoint_scene.render()
        return np.array(rendered_keypoint_pil)


class VisualizationDispatcher(metaclass=Singleton):
    """Dispatcher for creating visualizations."""

    def __init__(self) -> None:
        self._creator_map = {
            DetectionResult: DetectionVisualizerCreator(),
            ClassificationResult: ClassificationVisualizerCreator(),
            InstanceSegmentationResult: InstanceSegmentationVisualizerCreator(),
            AnomalyResult: AnomalyDetectionVisualizerCreator(),
            ImageResultWithSoftPrediction: SegmentationVisualizerCreator(),
            DetectedKeypoints: KeypointVisualizerCreator(),
        }

    def create_visualization(self, original_image: np.ndarray, predictions: Result) -> np.ndarray | None:
        """Create a visualization of the predictions on the original image."""
        if original_image is None or original_image.size == 0:
            raise ValueError("The original_image parameter must not be None or empty.")

        creator = self._creator_map.get(type(predictions))
        if creator is not None:
            return creator.create_visualization(original_image, predictions)
        logger.error(f"Visualization for {type(predictions)} is not suppported.")
        return None


class Visualizer:
    @staticmethod
    def overlay_predictions(original_image: np.ndarray, predictions: Result) -> np.ndarray:
        """Overlay predictions on the original image based on the type of predictions."""
        try:
            visualization = VisualizationDispatcher().create_visualization(original_image, predictions)
            if visualization is None:
                # If no visualization could be created, return the original image
                return original_image
        except Exception as e:
            logger.exception("An error occurred while creating visualization, returning original image.", exc_info=e)
            return original_image
        return visualization
