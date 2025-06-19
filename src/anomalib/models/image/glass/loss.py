# Original Code
# Copyright (c) 2021 @Hsuxu
# https://github.com/Hsuxu/Loss_ToolBox-PyTorch.
# SPDX-License-Identifier: Apache-2.0
#
# Modified
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import torch
from torch import nn


class FocalLoss(nn.Module):
    """Implementation of Focal Loss with support for smoothed label cross-entropy, as proposed in
    'Focal Loss for Dense Object Detection' (https://arxiv.org/abs/1708.02002).

    The focal loss formula is:
        Focal_Loss = -1 * alpha * (1 - pt) ** gamma * log(pt)

    Args:
        num_class (int): Number of classes.
        alpha (float or Tensor): Scalar or Tensor weight factor for class imbalance. If float, `balance_index` should be set.
        gamma (float): Focusing parameter that reduces the relative loss for well-classified examples (gamma > 0).
        smooth (float): Label smoothing factor for cross-entropy.
        balance_index (int): Index of the class to balance when `alpha` is a float.
        size_average (bool, optional): If True (default), the loss is averaged over the batch; otherwise, the loss is summed.

    """

    def __init__(
        self,
        apply_nonlin: nn.Module | None = None,
        alpha: float | torch.Tensor = None,
        gamma: float = 2,
        balance_index: int = 0,
        smooth: float = 1e-5,
        size_average: bool = True,
    ):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)

        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError("Not support alpha type")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key,
                self.smooth / (num_class - 1),
                1.0 - self.smooth,
            )
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
