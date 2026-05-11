# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Vision Foundation Model (VFM) based zero-shot anomaly detection model.

Example:
    >>> from anomalib.models.image import AnomalyVFM
    >>> # Zero-shot approach
    >>> model = AnomalyVFM()  # doctest: +SKIP

"""

import logging

from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchvision.transforms.v2 import Compose, Resize

from anomalib import LearningType, PrecisionType
from anomalib.data import ImageBatch, InferenceBatch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import AnomalyVFMModel

logger = logging.getLogger(__name__)


DEFAULT_IMAGE_SIZE = 768


class AnomalyVFM(AnomalibModule):
    """Vision Foundation Model (VFM) based zero-shot anomaly detection model.

    Example:
        >>> from anomalib.models.image import AnomalyVFM
        >>> # Zero-shot approach
        >>> model = AnomalyVFM()  # doctest: +SKIP

    """

    def __init__(
        self,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
        precision: str | PrecisionType = PrecisionType.FLOAT32,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        self.model = AnomalyVFMModel()

        if isinstance(precision, str):
            self.model.precision = PrecisionType(precision.lower())
        else:
            self.model.precision = PrecisionType.FLOAT32

    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Configure the default pre-processor for AnomalyVFM.

        Pre-processor resizes images.

        Args:
            image_size (tuple[int, int] | None, optional): Target size for
                resizing. Defaults to ``(768, 768)``.

        Returns:
            PreProcessor: Configured AnomalyVFM pre-processor
        """
        image_size = image_size or (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
        return PreProcessor(
            transform=Compose([
                Resize(image_size, antialias=True),
            ]),
        )

    def validation_step(self, batch: ImageBatch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step and return the anomaly map and anomaly score.

        Args:
            batch (ImageBatch): Input batch
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT | None: batch dictionary containing anomaly-maps and anomaly-scores.
        """
        # These variables are not used.
        del args, kwargs

        # Get anomaly maps and predicted scores from the model.
        anomaly_scores, anomaly_maps = self.model(batch.image)
        predictions = InferenceBatch(pred_score=anomaly_scores, anomaly_map=anomaly_maps)

        return batch.update(**predictions._asdict())

    def test_step(self, batch: ImageBatch, *args, **kwargs) -> ImageBatch:  # type: ignore[override]
        """Redirect to validation step."""
        return self.validation_step(batch, *args, **kwargs)

    def predict_step(self, batch: ImageBatch, *args, **kwargs) -> ImageBatch:  # type: ignore[override]
        """Redirect to validation step."""
        return self.validation_step(batch, *args, **kwargs)

    @property
    def learning_type(self) -> LearningType:
        """Get the learning type of the model. This model always uses zero-shot learning.

        Returns:
            LearningType: ZERO_SHOT.
        """
        return LearningType.ZERO_SHOT

    @property
    def trainer_arguments(self) -> dict[str, int | float]:
        """Get trainer arguments.

        Returns:
            dict[str, int | float]: Empty dict as no training is needed.
        """
        return {}

    @staticmethod
    def configure_transforms(image_size: tuple[int, int] | None = None) -> None:
        """Configure image transforms.

        Args:
            image_size (tuple[int, int] | None, optional): Ignored as each model
                has its own transforms. Defaults to None.
        """
        if image_size is not None:
            logger.warning("Ignoring image_size argument as each model has its own transforms.")

    @classmethod
    def configure_post_processor(cls) -> PostProcessor | None:
        """Configure the default post processor.

        Returns:
            PostProcessor: Post-processor for one-class models that
                converts raw scores to anomaly predictions
        """
        return PostProcessor()

    @staticmethod
    def _export_not_supported_message() -> None:
        logger.warning("Exporting the model is not supported for AnomalyVFM model. Skipping...")

    def to_torch(self, *_, **__) -> None:  # type: ignore[override]
        """Skip export to torch."""
        return self._export_not_supported_message()

    def to_onnx(self, *_, **__) -> None:  # type: ignore[override]
        """Skip export to onnx."""
        return self._export_not_supported_message()

    def to_openvino(self, *_, **__) -> None:  # type: ignore[override]
        """Skip export to openvino."""
        return self._export_not_supported_message()
