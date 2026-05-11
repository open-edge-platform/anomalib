# Original Code
# Copyright (c) 2021 @Hsuxu
# https://github.com/Hsuxu/Loss_ToolBox-PyTorch.
# SPDX-License-Identifier: Apache-2.0
#
# Modified
# Copyright (C) 2026 Intel Corporation
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

Returns:
    torch.Tensor: Scalar loss value (averaged or summed based on `size_average`).

Raises:
    ValueError: If `smooth` is outside the range [0, 1].
    TypeError: If `alpha` is not a supported type.
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
        apply_nonlinearity (nn.Module | None): Optional non-linearity to apply to logits before loss computation
            (e.g., ``nn.Softmax(dim=1)`` or ``nn.Sigmoid()``). Defaults to ``None``.
        alpha (float | torch.Tensor | np.ndarray | None): Weighting factor for class imbalance. Can be:
            - ``None``: Equal weighting for all classes.
            - ``float``: Class at ``balance_index`` is weighted by ``alpha``, others by ``1 - alpha``.
            - ``Tensor`` or array: Direct per-class weights of shape ``(num_classes,)``.
            Defaults to ``None``.
        gamma (float): Focusing parameter (> 0) to reduce loss contribution from easy examples. Defaults to ``2``.
        balance_index (int): Index of the class to apply ``alpha`` to when ``alpha`` is a float. Defaults to ``0``.
        smooth (float): Label smoothing factor in ``[0, 1]``. Defaults to ``1e-5``.
        size_average (bool): If ``True``, average the loss over the batch; otherwise sum. Defaults to ``True``.
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
        elif isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.view(num_classes, 1).float()
            alpha = alpha / alpha.sum()
        elif isinstance(self.alpha, (list | np.ndarray)):
            alpha = torch.FloatTensor(alpha).view(num_classes, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(num_classes, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            msg = f"Unsupported alpha type: {type(self.alpha)}"
            raise TypeError(msg)

        if alpha.device != logits.device:
            alpha = alpha.to(logits.device)

        idx = target.to(device=logits.device, dtype=torch.long)
        one_hot_key = torch.zeros(target.size(0), num_classes, device=logits.device, dtype=logits.dtype)
        one_hot_key.scatter_(1, idx, 1)

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
