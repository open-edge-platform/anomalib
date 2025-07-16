"""Loss functions for UniNet."""

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
from torch.nn import functional


class UniNetLoss(nn.Module):
    """Loss function for UniNet.

    Args:
        lambda_weight (float): Hyperparameter for balancing the loss. Defaults to 0.7.
        temperature (float): Temperature for contrastive learning. Defaults to 2.0.
    """

    def __init__(self, lambda_weight: float = 0.7, temperature: float = 2.0) -> None:
        super().__init__()
        self.lambda_weight = lambda_weight
        self.temperature = temperature

    def forward(
        self,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
        margin: int = 1,
        mask: torch.Tensor | None = None,
        stop_gradient: bool = False,
    ) -> torch.Tensor:
        """Compute the loss.

        Args:
            student_features (list[torch.Tensor]): Student features.
            teacher_features (list[torch.Tensor]): Teacher features.
            margin (int): Hyperparameter for controlling the boundary.
            mask (torch.Tensor | None): Mask for the prediction. Mask is of shape Bx1xHxW
            stop_gradient (bool): Whether to stop the gradient into teacher features.
        """
        loss = 0.0
        margin_loss_a = 0.0

        for idx in range(len(student_features)):
            student_feature = student_features[idx]
            teacher_feature = teacher_features[idx].detach() if stop_gradient else teacher_features[idx]

            n, c, h, w = student_feature.shape
            student_feature = student_feature.view(n, c, -1).transpose(1, 2)  # (N, H+W, C)
            teacher_feature = teacher_feature.view(n, c, -1).transpose(1, 2)  # (N, H+W, C)

            student_feature_normalized = functional.normalize(student_feature, p=2, dim=2)
            teacher_feature_normalized = functional.normalize(teacher_feature, p=2, dim=2)

            cosine_loss = 1 - functional.cosine_similarity(
                student_feature_normalized,
                teacher_feature_normalized,
                dim=2,
            )
            cosine_loss = cosine_loss.mean()

            similarity = (
                torch.matmul(student_feature_normalized, teacher_feature_normalized.transpose(1, 2)) / self.temperature
            )
            similarity = torch.exp(similarity)
            similarity_sum = similarity.sum(dim=2, keepdim=True)
            similarity = similarity / (similarity_sum + 1e-8)
            diag_sum = torch.diagonal(similarity, dim1=1, dim2=2)

            # unsupervised and only normal (or abnormal)
            if mask is None:
                contrastive_loss = -torch.log(diag_sum + 1e-8).mean()
                margin_loss_n = functional.relu(margin - diag_sum).mean()

            # supervised
            else:
                # gt label
                if len(mask.shape) < 3:
                    normal_mask = mask == 0
                    abnormal_mask = mask == 1
                # gt mask
                else:
                    mask_ = functional.interpolate(mask, size=(h, w), mode="nearest").squeeze(1)
                    mask_flat = mask_.view(mask_.size(0), -1)

                    normal_mask = mask_flat == 0
                    abnormal_mask = mask_flat == 1

                if normal_mask.sum() > 0:
                    diag_sim_normal = diag_sum[normal_mask]
                    contrastive_loss = -torch.log(diag_sim_normal + 1e-8).mean()
                    margin_loss_n = functional.relu(margin - diag_sim_normal).mean()

                if abnormal_mask.sum() > 0:
                    diag_sim_abnormal = diag_sum[abnormal_mask]
                    margin_loss_a = functional.relu(diag_sim_abnormal - margin / 2).mean()

            margin_loss = margin_loss_n + margin_loss_a

            loss += cosine_loss * self.lambda_weight + contrastive_loss * (1 - self.lambda_weight) + margin_loss

        return loss
