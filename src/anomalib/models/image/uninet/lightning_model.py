"""Lightning model for UniNet."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy

import torch
import torchvision
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional
from torchvision.models.feature_extraction import create_feature_extractor

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
        student_backbone: str = "wide_resent50_2",
        teacher_backbone: str = "wide_resent50_2",
    ) -> None:
        self.dfs = DomainRelatedFeatureSelection()
        self.loss = UniNetLoss(temperature=2.0)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        source_teacher = self._get_teacher(teacher_backbone)
        target_teacher = copy.deepcopy(source_teacher)
        self.model = UniNetModel(
            student=get_decoder(student_backbone),
            bottleneck=BottleneckLayer(block=AttentionBottleneck, layers=3),
            source_teacher=source_teacher,
            target_teacher=target_teacher,
        )

    def training_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a training step of UniNet."""
        del args, kwargs  # These variables are not used.
        source_target_features, student_features, predictions = self.model(batch.image)
        student_features = self._feature_selection(source_target_features, student_features)
        loss = self._compute_loss(student_features, source_target_features, predictions, batch.gt_label, batch.gt_mask)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a validation step of UniNet."""
        del args, kwargs  # These variables are not used.
        if batch.image is None or not isinstance(batch.image, torch.Tensor):
            msg = "Expected batch.image to be a tensor, but got None or non-tensor type"
            raise ValueError(msg)

        source_target_features, student_features, _ = self.model(batch.image)
        output_list: list[list[torch.Tensor]] = [[] for _ in range(self.model.num_teachers * 3)]
        for idx, (target_feature, student_feature) in enumerate(
            zip(source_target_features, student_features, strict=False),
        ):
            output = 1 - functional.cosine_similarity(target_feature, student_feature)
            output_list[idx].append(output)

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
                self.model.student.parameters() + self.model.bottleneck.parameters() + self.dfs.parameters(),
                lr=5e-3,
                betas=(0.9, 0.999),
                weight_decay=1e-5,
            ),
            torch.optim.AdamW(self.target_teacher.parameters(), lr=1e-6, betas=(0.9, 0.999), weight_decay=1e-5),
        )

    @staticmethod
    def _get_teacher(backbone: str) -> nn.Module:
        """Get the teacher model.

        In the original code, the teacher resnet model is used to extract features from the input image.
        We can just use the feature extractor from torchvision to extract the features.

        Args:
            backbone (str): The backbone model to use.

        Returns:
            nn.Module: The teacher model.
        """
        model = getattr(torchvision.models, backbone)(pretrained=True)
        return create_feature_extractor(model, return_nodes=["layer1", "layer2", "layer3"])

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
            selected_features = domain_related_feature_selection(source_features, target_features, maximize=maximize)
        return selected_features

    def _compute_loss(
        self,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
        predictions: torch.Tensor | None = None,
        label: float | None = None,
        mask: torch.Tensor | None = None,
        stop_gradient: bool = False,
    ) -> torch.Tensor:
        """Compute the loss.

        Args:
            student_features (list[torch.Tensor]): Student features.
            teacher_features (list[torch.Tensor]): Teacher features.
            predictions (torch.Tensor): Predictions.
            label (float | None): Label for the prediction.
            mask (torch.Tensor | None): Mask for the prediction.
            stop_gradient (bool): Whether to stop the gradient into teacher features.
        """
        # TODO(ashwinvaidya17): figure out the dimension of predictions and update the docstring
        if mask is not None:
            mask_ = mask
        else:
            assert label is not None, "Label is required when mask is not provided"
            mask_ = label

        loss = self.loss(student_features, teacher_features, mask=mask_, stop_gradient=stop_gradient)
        if predictions is not None and label is not None:
            loss += self.bce_loss(predictions[0], label) + self.bce_loss(predictions[1], label)

        return loss
