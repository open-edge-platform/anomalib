# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Vision Foundation Model (VFM) based zero-shot anomaly detection model.

Example:
    >>> from anomalib.models.image import AnomalyVFM
    >>> # Zero-shot approach
    >>> model = AnomalyVFM()  # doctest: +SKIP

"""

import logging

from huggingface_hub import hf_hub_download
from lightning.pytorch.utilities.types import STEP_OUTPUT
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional

from anomalib import LearningType
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
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        self.model = AnomalyVFMModel()
        weights_path = hf_hub_download(
            repo_id="MaticFuc/anomalyvfm_radio",
            filename="model.safetensors",
        )
        safe_state_dict = load_file(weights_path)
        self.model.load_state_dict(safe_state_dict)
        self.mean_kernel = nn.AvgPool2d((5, 5), 1, 5 // 2)
        self.pre_processor = PreProcessor(transform=self.model.model.get_img_transform())

    def validation_step(self, batch: ImageBatch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform the validation step and return the anomaly map and anomaly score.

        Args:
            batch (dict[str, str | torch.Tensor]): Input batch
            args: Additional arguments.
            kwargs: Additional keyword arguments.

        Returns:
            STEP_OUTPUT | None: batch dictionary containing anomaly-maps and anomaly-scores.
        """
        # These variables are not used.
        del args, kwargs

        # Get anomaly maps and predicted scores from the model.
        anomaly_scores, anomaly_maps = self.model(batch.image)
        anomaly_maps = self.mean_kernel(anomaly_maps)
        anomaly_maps = functional.interpolate(
            anomaly_maps,
            size=self.model.model.H,
            mode="bilinear",
            align_corners=False,
        )
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
        """Get the learning type of the model.

        Returns:
            LearningType: ZERO_SHOT if k_shot=0, else FEW_SHOT.
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
