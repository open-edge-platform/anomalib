# Original Code
# Copyright (c) 2025 Shun Wei
# https://github.com/pangdatangtt/UniNet
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model for UniNet.

See Also:
    :class:`anomalib.models.image.uninet.lightning_model.UniNet`:
        UniNet Lightning model.
"""

import torch
import torchvision
from torch import nn
from torch.fx import GraphModule
from torch.nn import functional as F  # noqa: N812
from torchvision.models.feature_extraction import create_feature_extractor

from anomalib.data import InferenceBatch
from anomalib.models.components.backbone import get_decoder

from .components import AttentionBottleneck, BottleneckLayer, DomainRelatedFeatureSelection, weighted_decision_mechanism


class UniNetModel(nn.Module):
    """UniNet PyTorch model.

    It consists of teachers, student, and bottleneck modules.

    Args:
        student_backbone (str): Student backbone model.
        teacher_backbone (str): Teacher backbone model.
        loss (nn.Module): Loss function.
    """

    def __init__(
        self,
        student_backbone: str,
        teacher_backbone: str,
        loss: nn.Module,
    ) -> None:
        super().__init__()
        self.teachers = Teachers(teacher_backbone)
        self.student = get_decoder(student_backbone)
        self.bottleneck = BottleneckLayer(block=AttentionBottleneck, layers=3)
        self.dfs = DomainRelatedFeatureSelection()

        self.loss = loss
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        # Used to post-process the student features from the de_resnet model to get the predictions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1)

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | InferenceBatch:
        """Forward pass of the UniNet model.

        Args:
            images (torch.Tensor): Input images.
            masks (torch.Tensor | None): Ground truth masks.
            labels (torch.Tensor | None): Ground truth labels.

        Returns:
            torch.Tensor | InferenceBatch: Loss or InferenceBatch.
        """
        source_target_features, bottleneck_inputs = self.teachers(images)
        bottleneck_outputs = self.bottleneck(bottleneck_inputs)

        student_features = self.student(bottleneck_outputs)

        # These predictions are part of the de_resnet model of the original code.
        # since we are using the de_resnet model from anomalib, we need to compute predictions here
        predictions = self.avgpool(student_features[0])
        predictions = torch.flatten(predictions, 1)
        predictions = self.fc(predictions).squeeze()
        predictions = predictions.chunk(dim=0, chunks=2)

        student_features = [d.chunk(dim=0, chunks=2) for d in student_features]
        student_features = [
            student_features[0][0],
            student_features[1][0],
            student_features[2][0],
            student_features[0][1],
            student_features[1][1],
            student_features[2][1],
        ]
        if self.training:
            student_features = self._feature_selection(source_target_features, student_features)
            return self._compute_loss(
                student_features,
                source_target_features,
                predictions,
                labels,
                masks,
            )

        output_list: list[torch.Tensor] = []
        for target_feature, student_feature in zip(source_target_features, student_features, strict=True):
            output = 1 - F.cosine_similarity(target_feature, student_feature)  # B*64*64
            output_list.append(output)

        anomaly_score, anomaly_map = weighted_decision_mechanism(
            batch_size=images.shape[0],
            output_list=output_list,
            alpha=0.01,
            beta=3e-05,
            output_size=images.shape[-2:],
        )
        return InferenceBatch(
            pred_score=anomaly_score,
            anomaly_map=anomaly_map,
        )

    def _compute_loss(
        self,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
        predictions: tuple[torch.Tensor, torch.Tensor] | None = None,
        label: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        stop_gradient: bool = False,
    ) -> torch.Tensor:
        """Compute the loss.

        Args:
            student_features (list[torch.Tensor]): Student features.
            teacher_features (list[torch.Tensor]): Teacher features.
            predictions (tuple[torch.Tensor, torch.Tensor] | None): Predictions is (B, B)
            label (torch.Tensor | None): Label for the prediction.
            mask (torch.Tensor | None): Mask for the prediction. Mask is of shape BxHxW
            stop_gradient (bool): Whether to stop the gradient into teacher features.

        Returns:
            torch.Tensor: Loss.
        """
        if mask is not None:
            mask_ = mask.float().unsqueeze(1)  # Bx1xHxW
        else:
            assert label is not None, "Label is required when mask is not provided"
            mask_ = label.float()

        loss = self.loss(student_features, teacher_features, mask=mask_, stop_gradient=stop_gradient)
        if predictions is not None and label is not None:
            loss += self.bce_loss(predictions[0], label.float()) + self.bce_loss(predictions[1], label.float())

        return loss

    def _feature_selection(
        self,
        target_features: list[torch.Tensor],
        source_features: list[torch.Tensor],
        maximize: bool = True,
    ) -> list[torch.Tensor]:
        """Feature selection.

        Args:
            source_features (list[torch.Tensor]): Source features.
            target_features (list[torch.Tensor]): Target features.
            maximize (bool): Used for weights computation. If True, the weights are computed by subtracting the
                max value from the target feature. Defaults to True.
        """
        return self.dfs(source_features, target_features, maximize=maximize)


class Teachers(nn.Module):
    """Teachers module for UniNet.

    Args:
        source_teacher (nn.Module): Source teacher model.
        target_teacher (nn.Module | None): Target teacher model.
    """

    def __init__(self, teacher_backbone: str) -> None:
        super().__init__()
        self.source_teacher = self._get_teacher(teacher_backbone).eval()
        self.target_teacher = self._get_teacher(teacher_backbone)

    @staticmethod
    def _get_teacher(backbone: str) -> GraphModule:
        """Get the teacher model.

        In the original code, the teacher resnet model is used to extract features from the input image.
        We can just use the feature extractor from torchvision to extract the features.

        Args:
            backbone (str): The backbone model to use.

        Returns:
            GraphModule: The teacher model.
        """
        model = getattr(torchvision.models, backbone)(pretrained=True)
        return create_feature_extractor(model, return_nodes=["layer3", "layer2", "layer1"])

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass of the teachers.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]: Source features or source and target features.
        """
        with torch.no_grad():
            source_features = self.source_teacher(images)

        target_features = self.target_teacher(images)

        bottleneck_inputs = [
            torch.cat([a, b], dim=0) for a, b in zip(target_features.values(), source_features.values(), strict=True)
        ]  # 512, 1024, 2048

        return list(source_features.values()) + list(target_features.values()), bottleneck_inputs
