"""PyTorch model for UniNet.

See Also:
    :class:`anomalib.models.image.uninet.lightning_model.UniNet`:
        UniNet Lightning model.
"""

# Original Code
# Copyright (c) 2025 Shun Wei
# https://github.com/pangdatangtt/UniNet
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torchvision
from torch import nn
from torch.fx import GraphModule
from torchvision.models.feature_extraction import create_feature_extractor


class UniNetModel(nn.Module):
    """UniNet PyTorch model.

    It consists of teachers, student, and bottleneck modules.

    Args:
        student (nn.Module): Student model.
        bottleneck (nn.Module): Bottleneck model.
        source_teacher (nn.Module): Source teacher model.
        target_teacher (nn.Module | None): Target teacher model.
    """

    def __init__(
        self,
        student: nn.Module,
        bottleneck: nn.Module,
        teacher_backbone: str,
    ) -> None:
        super().__init__()
        self.num_teachers = 2  # in the original code, there is an option to have only one teacher
        self.teachers = Teachers(teacher_backbone)
        self.student = student
        self.bottleneck = bottleneck

        # Used to post-process the student features from the de_resnet model to get the predictions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the UniNet model.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: Source target features,
                student features, and predictions.
        """
        source_target_features, bottleneck_inputs = self.teachers(images)
        bottleneck_outputs = self.bottleneck(bottleneck_inputs)

        student_features = self.student(bottleneck_outputs)

        # These predictions are part of the de_resnet model of the original code.
        # since we are using the de_resnet model from anomalib, we need to compute predictions here
        predictions = self.avgpool(student_features[0])
        predictions = torch.flatten(predictions, 1)
        predictions = self.fc(predictions).squeeze()

        student_features = [d.chunk(dim=0, chunks=2) for d in student_features]
        student_features = [
            student_features[0][0],
            student_features[1][0],
            student_features[2][0],
            student_features[0][1],
            student_features[1][1],
            student_features[2][1],
        ]

        predictions = predictions.chunk(dim=0, chunks=2)

        return source_target_features, student_features, predictions


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
        # TODO(ashwinvaidya17): revisit this method and clean up multiple return statements
        with torch.no_grad():
            source_features = self.source_teacher(images)

        target_features = self.target_teacher(images)

        bottleneck_inputs = [
            torch.cat([a, b], dim=0) for a, b in zip(target_features.values(), source_features.values(), strict=True)
        ]  # 512, 1024, 2048

        return list(source_features.values()) + list(target_features.values()), bottleneck_inputs
