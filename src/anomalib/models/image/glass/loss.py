# Original Code
# Copyright (c) 2021 @Hsuxu
# https://github.com/Hsuxu/Loss_ToolBox-PyTorch.
# SPDX-License-Identifier: Apache-2.0
#
# Modified
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Focal Loss for multi-class classification with optional label smoothing and class weighting.

This loss function is designed to address class imbalance by down-weighting easy examples and focusing training
on hard, misclassified examples. It is based on the paper:
"Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002).

The focal loss formula is:
    FL(pt) = -alpha * (1 - pt) ** gamma * log(pt)

where:
    - pt is the predicted probability of the correct class
    - alpha is a class balancing factor
    - gamma is a focusing parameter

Supports optional label smoothing and flexible alpha input (scalar or per-class tensor). Can be used with raw logits,
applying a specified non-linearity (e.g., softmax or sigmoid).

Args:
    apply_nonlinearity (nn.Module or None): Optional non-linearity to apply to the logits before loss computation.
        For example, use `nn.Softmax(dim=1)` or `nn.Sigmoid()` if logits are not normalized.
    alpha (float or torch.Tensor, optional): Class balancing factor. Can be:
        - None: Equal weighting for all classes.
        - float: Scalar for binary class weighting; applied to `balance_index`.
        - Tensor: Per-class weights of shape (num_classes,).
    gamma (float): Focusing parameter (> 0) to reduce the loss contribution from easy examples. Default is 2.
    balance_index (int): Index of the class to apply `alpha` to when `alpha` is a float.
    smooth (float): Label smoothing factor. A small value (e.g., 1e-5) helps prevent overconfidence.
    size_average (bool): If True, average the loss over the batch; if False, sum the loss.

Raises:
    ValueError: If `smooth` is outside the range [0, 1].
    TypeError: If `alpha` is not a supported type.

Inputs:
    logit (torch.Tensor): Raw model outputs (logits) of shape (B, C, ...) where B is batch size and C is number of
      classes.
    target (torch.Tensor): Ground-truth class indices of shape (B, 1, ...) or broadcastable to match logit.

Returns:
    torch.Tensor: Scalar loss value (averaged or summed based on `size_average`).
"""

import numpy as np
import torch
from torch import nn


class FocalLoss(nn.Module):
    """Implementation of Focal Loss with support for smoothed label cross-entropy.

    As proposed in 'Focal Loss for Dense Object Detection' (https://arxiv.org/abs/1708.02002).
    The focal loss formula is:
        Focal_Loss = -1 * alpha * (1 - pt) ** gamma * log(pt)

    Args:
        num_class (int): Number of classes.
        alpha (float or Tensor): Scalar or Tensor weight factor for class imbalance. If float, `balance_index` should be
          set.
        gamma (float): Focusing parameter that reduces the relative loss for well-classified examples (gamma > 0).
        smooth (float): Label smoothing factor for cross-entropy.
        balance_index (int): Index of the class to balance when `alpha` is a float.
        size_average (bool, optional): If True (default), the loss is averaged over the batch; otherwise, the loss is
          summed.
    """

    def __init__(
        self,
        apply_nonlinearity: nn.Module | None = None,
        alpha: float | torch.Tensor | np.ndarray | None = None,
        gamma: float = 2,
        balance_index: int = 0,
        smooth: float = 1e-5,
        size_average: bool = True,
    ) -> None:
        """Initializes the FocalLoss instance.

        Args:
            apply_nonlinearity (nn.Module or None): Optional non-linearity to apply to logits (e.g., softmax or sigmoid)
            alpha (float or torch.Tensor, optional): Weighting factor for class imbalance. Can be:
                - None: Equal weighting.
                - float: Class at `balance_index` is weighted by `alpha`, others by 1 - `alpha`.
                - Tensor: Direct per-class weights.
            gamma (float): Focusing parameter for down-weighting easy examples (y > 0).
            balance_index (int): Index of the class to apply `alpha` to when `alpha` is a float.
            smooth (float): Label smoothing factor (0 to 1).
            size_average (bool): If True, average the loss over the batch. If False, sum the loss.
        """
        super().__init__()
        self.apply_nonlinearity = apply_nonlinearity
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None and (self.smooth < 0 or self.smooth > 1.0):
            msg = "smooth value should be in [0,1]"
            raise ValueError(msg)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the focal loss between `logit` predictions and ground-truth `target`.

        Args:
            logits (torch.Tensor): The predicted logits of shape (B, C, ...) where B is batch size and C is the
              number of classes.
            target (torch.Tensor): The ground-truth class indices of shape (B, 1, ...) or broadcastable to logit.

        Returns:
            torch.Tensor: Computed focal loss value (averaged or summed depending on `size_average`).
        """
        if self.apply_nonlinearity is not None:
            logits = self.apply_nonlinearity(logits)
        num_classes = logits.shape[1]

        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)
            logits = logits.permute(0, 2, 1).contiguous()
            logits = logits.view(-1, logits.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)

        alpha = self.alpha
        if self.alpha is None:
            alpha = torch.ones(num_classes, 1)
        elif isinstance(self.alpha, (list | np.ndarray)):
            alpha = torch.FloatTensor(alpha).view(num_classes, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(num_classes, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            msg = "Not support alpha type"
            raise TypeError(msg)

        if alpha.device != logits.device:
            alpha = alpha.to(logits.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), num_classes).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logits.device:
            one_hot_key = one_hot_key.to(logits.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key,
                self.smooth / (num_classes - 1),
                1.0 - self.smooth,
            )
        pt = (one_hot_key * logits).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma
        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        return loss.mean() if self.size_average else loss.sum()
