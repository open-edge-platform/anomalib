"""Lightning model for UniNet."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.nn import functional

from anomalib import LearningType
from anomalib.data import Batch, InferenceBatch
from anomalib.models.components import AnomalibModule
from anomalib.models.components.backbone import get_decoder

from .anomaly_map import weighted_decision_mechanism
from .components import (
    AttentionBottleneck,
    BottleneckLayer,
    DomainRelatedFeatureSelection,
    domain_related_feature_selection,
)
from .loss import UniNetLoss
from .torch_model import UniNetModel


class UniNet(AnomalibModule):
    """UniNet model for anomaly detection.

    Args:
        student_backbone (str): The backbone model to use for the student.
        teacher_backbone (str): The backbone model to use for the teacher.
    """

    def __init__(
        self,
        student_backbone: str = "wide_resnet50_2",
        teacher_backbone: str = "wide_resnet50_2",
    ) -> None:
        super().__init__()
        self.dfs = DomainRelatedFeatureSelection()
        self.loss = UniNetLoss(temperature=2.0)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.model = UniNetModel(
            student=get_decoder(student_backbone),
            bottleneck=BottleneckLayer(block=AttentionBottleneck, layers=3),
            teacher_backbone=teacher_backbone,
        )
        self.automatic_optimization = False

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a training step of UniNet."""
        del args, kwargs  # These variables are not used.
        assert batch.gt_label is not None, "Ground truth label is required for training"

        source_target_features, student_features, predictions = self.model(batch.image)
        student_features = self._feature_selection(source_target_features, student_features)
        loss = self._compute_loss(
            student_features,
            source_target_features,
            predictions,
            batch.gt_label.float(),
            batch.gt_mask,
        )

        optimizer_1, optimizer_2 = self.optimizers()
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        self.manual_backward(loss)
        optimizer_1.step()
        optimizer_2.step()

        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a validation step of UniNet."""
        del args, kwargs  # These variables are not used.
        if batch.image is None or not isinstance(batch.image, torch.Tensor):
            msg = "Expected batch.image to be a tensor, but got None or non-tensor type"
            raise ValueError(msg)

        target_features, student_features, _ = self.model(batch.image)
        output_list: list[torch.Tensor] = []
        for target_feature, student_feature in zip(target_features, student_features, strict=True):
            output = 1 - functional.cosine_similarity(target_feature, student_feature)  # B*64*64
            output_list.append(output)

        anomaly_score, anomaly_map = weighted_decision_mechanism(
            batch_size=batch.image.shape[0],
            output_list=output_list,
            alpha=0.01,
            beta=0.00003,
        )
        predictions = InferenceBatch(
            pred_score=anomaly_score,
            anomaly_map=anomaly_map,
        )
        return batch.update(**predictions._asdict())

    def configure_optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """Configure optimizers for training.

        Returns:
            tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Optimizers for student and target teacher.
        """
        # TODO(ashwinvaidya17): refactor this
        return (
            torch.optim.AdamW(
                list(self.model.student.parameters())
                + list(self.model.bottleneck.parameters())
                + list(self.dfs.parameters()),
                lr=5e-3,
                betas=(0.9, 0.999),
                weight_decay=1e-5,
            ),
            torch.optim.AdamW(
                self.model.teachers.target_teacher.parameters(),
                lr=1e-6,
                betas=(0.9, 0.999),
                weight_decay=1e-5,
            ),
        )

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Does not require any trainer arguments."""
        return {}

    @property
    def learning_type(self) -> LearningType:
        """The model uses one-class learning.

        Though technicaly it suppports multi-class and few_shot as well.
        """
        return LearningType.ONE_CLASS

    def _feature_selection(
        self,
        source_features: list[torch.Tensor],
        target_features: list[torch.Tensor],
        maximize: bool = True,
    ) -> list[torch.Tensor]:
        """Feature selection.

        Args:
            source_features (list[torch.Tensor]): Source features.
            target_features (list[torch.Tensor]): Target features.
            maximize (bool): Used for weights computation. If True, the weights are computed by subtracting the
                max value from the target feature. Defaults to True.
        """
        if self.dfs is not None:
            selected_features = self.dfs(source_features, target_features, maximize=maximize)
        else:
            # TODO(ashwinvaidya17): self.dfs will never be none with current implementation
            selected_features = domain_related_feature_selection(source_features, target_features, maximize=maximize)
        return selected_features

    def _compute_loss(
        self,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
        predictions: tuple[torch.Tensor, torch.Tensor] | None = None,
        label: float | None = None,
        mask: torch.Tensor | None = None,
        stop_gradient: bool = False,
    ) -> torch.Tensor:
        """Compute the loss.

        Args:
            student_features (list[torch.Tensor]): Student features.
            teacher_features (list[torch.Tensor]): Teacher features.
            predictions (tuple[torch.Tensor, torch.Tensor] | None): Predictions is (B, B)
            label (float | None): Label for the prediction.
            mask (torch.Tensor | None): Mask for the prediction. Mask is of shape BxHxW
            stop_gradient (bool): Whether to stop the gradient into teacher features.

        Returns:
            torch.Tensor: Loss.
        """
        if mask is not None:
            mask_ = mask.unsqueeze(1)  # Bx1xHxW
        else:
            assert label is not None, "Label is required when mask is not provided"
            mask_ = label

        loss = self.loss(student_features, teacher_features, mask=mask_, stop_gradient=stop_gradient)
        if predictions is not None and label is not None:
            loss += self.bce_loss(predictions[0], label) + self.bce_loss(predictions[1], label)

        return loss
