"""Lightning model for UniNet."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib.data import Batch, InferenceBatch
from anomalib.models.components import AnomalibModule

from .anomaly_map import weighted_decision_mechanism
from .dfs import DomainRelatedFeatureSelection, domain_related_feature_selection
from .loss import UniNetLoss
from .torch_model import UniNetModel
from torchvision.models.resnet 


class UniNet(AnomalibModule):
    """UniNet model for anomaly detection.

    Args:
        dfs (DomainRelatedFeatureSelection | None): Domain related feature selection module.
    """

    def __init__(self, dfs: DomainRelatedFeatureSelection | None = None) -> None:
        # TODO: maybe DFS shouldn't be passed as argument as not sure when it will be the case it needs to be passed
        self.dfs = dfs
        self.loss = UniNetLoss(temperature=2.0)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.model = UniNetModel(student=, bottleneck=, source_teacher=, target_teacher=)

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
        source_target_features, student_features, _ = self.model(batch.image)
        output_list = [[] for _ in range(self.model.num_teachers * 3)]
        for idx, (target_feature, student_feature) in enumerate(
            zip(source_target_features, student_features, strict=False),
        ):
            output = 1 - F.cosine_similarity(target_feature, student_feature)
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
        # TODO: refactor this
        return [
            torch.optim.AdamW(
                self.model.student.parameters() + self.model.bottleneck.parameters() + self.dfs.parameters(),
                lr=5e-3,
                betas=(0.9, 0.999),
                weight_decay=1e-5,
            ),
            torch.optim.AdamW(self.target_teacher.parameters(), lr=1e-6, betas=(0.9, 0.999), weight_decay=1e-5),
        ]

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
        label: int | None = None,
        mask: torch.Tensor | None = None,
        stop_gradient: bool = False,
    ) -> torch.Tensor:
        """Compute the loss.

        Args:
            student_features (list[torch.Tensor]): Student features.
            teacher_features (list[torch.Tensor]): Teacher features.
            predictions (torch.Tensor): Predictions.
            label (int | None): Label for the prediction.
            mask (torch.Tensor | None): Mask for the prediction.
            stop_gradient (bool): Whether to stop the gradient into teacher features.
        """
        # TODO: figure out the dimension of predictions and update the docstring
        if mask is not None:
            mask_ = mask
        else:
            assert label is not None, "Label is required when mask is not provided"
            mask_ = label

        loss = self.loss(student_features, teacher_features, mask=mask_, stop_gradient=stop_gradient)
        if predictions is not None and label is not None:
            loss += self.bce_loss(predictions[0], label.float()) + self.bce_loss(predictions[1], label.float())

        return loss
