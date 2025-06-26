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
from torch import nn


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
        source_teacher: nn.Module,
        target_teacher: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.num_teachers = 1 if target_teacher is None else 2
        self.teachers = Teachers(source_teacher, target_teacher)
        self.student = student
        self.bottleneck = bottleneck

        # Used to post-process the student features from the de_resnet model to get the predictions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the UniNet model.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Source target features, student features, and predictions.
        """
        source_target_features, bottleneck_inputs = self.teachers(images)
        bottleneck_outputs = self.bottleneck(bottleneck_inputs)

        student_features = self.student(bottleneck_outputs)

        # These predictions are part of the de_resnet model of the original code.
        # since we are using the de_resnet model from anomalib, we need to compute predictions here
        predictions = self.avgpool(student_features[2])
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

    def __init__(self, source_teacher: nn.Module, target_teacher: nn.Module | None = None) -> None:
        super().__init__()
        self.source_teacher = source_teacher
        self.target_teacher = target_teacher

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

        if self.target_teacher is None:
            return source_features

        with torch.no_grad():
            target_features = self.target_teacher(images)
            bottleneck_inputs = [
                torch.cat([a, b], dim=0) for a, b in zip(target_features, source_features, strict=True)
            ]  # 512, 1024, 2048

            return source_features + target_features, bottleneck_inputs
