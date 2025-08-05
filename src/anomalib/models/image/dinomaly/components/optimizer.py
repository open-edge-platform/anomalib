# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Learning rate scheduler and optimizer for the Dinomaly model.

This module contains the WarmCosineScheduler and StableAdamW classes.
The code is based on the original dinomaly implementation:
https://github.com/guojiajeremy/Dinomaly/

"""

import math
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

ParamsT: TypeAlias = Iterable[torch.Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, torch.Tensor]]


class WarmCosineScheduler(_LRScheduler):
    """Cosine annealing scheduler with warmup.

    Learning rate scheduler that combines warm-up with cosine annealing.
    Reference: https://github.com/guojiajeremy/Dinomaly/blob/861a99b227fd2813b6ad8e8c703a7bea139ab735/utils.py#L775

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_value (float): Initial learning rate after warmup.
        final_value (float): Final learning rate after annealing.
        total_iters (int): Total number of iterations.
        warmup_iters (int, optional): Number of warmup iterations. Default is 0.
        start_warmup_value (float, optional): Starting learning rate for warmup. Default is 0.

    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_value: float,
        final_value: float,
        total_iters: int,
        warmup_iters: int = 0,
        start_warmup_value: float = 0,
    ) -> None:
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """Returns the learning rate for the current epoch.

        Returns:
            list[float]: List of learning rates for each parameter group.
        """
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]


class StableAdamW(Optimizer):
    """Implements "stable AdamW" algorithm with gradient clipping.

    This was introduced in "Stable and low-precision training for large-scale vision-language models".
    Publication Reference :  https://arxiv.org/abs/2304.13013
    Code reference (original implementation of the dinomaly model):
    https://github.com/guojiajeremy/Dinomaly/blob/861a99b227fd2813b6ad8e8c703a7bea139ab735/optimizers/StableAdamW.py#L10

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        clip_threshold: float = 1.0,
    ) -> None:
        if not lr >= 0.0:
            msg = f"Invalid learning rate: {lr}"
            raise ValueError(msg)
        if not eps >= 0.0:
            msg = f"Invalid epsilon value: {eps}"
            raise ValueError(msg)
        if not 0.0 <= betas[0] < 1.0:
            msg = f"Invalid beta parameter at index 0: {betas[0]}"
            raise ValueError(msg)
        if not 0.0 <= betas[1] < 1.0:
            msg = f"Invalid beta parameter at index 1: {betas[1]}"
            raise ValueError(msg)
        if not weight_decay >= 0.0:
            msg = f"Invalid weight decay value: {weight_decay}"
            raise ValueError(msg)
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "clip_threshold": clip_threshold,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restores the optimizer state from a given state dictionary.

        Ensures that the `amsgrad` parameter is set for each parameter group,
        maintaining compatibility when loading optimizer states from checkpoints.

        Args:
            state (dict[str, Any]): State dictionary to restore.
        """
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @staticmethod
    def _rms(tensor: torch.Tensor) -> float:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform step-weight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    msg = "StableAdamW does not support sparse gradients, please consider using SparseAdam instead."
                    raise RuntimeError(msg)
                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)  # , memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p)  # , memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exponential moving averages of squared gradients
                        state["max_exp_avg_sq"] = torch.zeros_like(p)  # , memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running average till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running average of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

                lr_scale = grad / denom
                lr_scale = max(1.0, self._rms(lr_scale) / group["clip_threshold"])

                step_size = group["lr"] / bias_correction1 / (lr_scale)

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
