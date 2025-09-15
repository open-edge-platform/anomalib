# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cosine Hard Mining Loss for training Dinomaly model.

The code is based on the `global_cosine_hm_percent()` method in the original dinomaly implementation
Reference: https://github.com/guojiajeremy/Dinomaly/blob/861a99b227fd2813b6ad8e8c703a7bea139ab735/utils.py#L70C5-L70C29
"""

from functools import partial

import torch


class CosineHardMiningLoss(torch.nn.Module):
    """Cosine similarity loss with hard mining for anomaly detection.

    This loss function implements a sophisticated training strategy for the Dinomaly model
    that prevents the decoder from becoming too effective at reconstructing anomalous regions.
    The key insight is to "loosen the point-by-point reconstruction constraint" by reducing
    the gradient contribution of well-reconstructed (easy) feature points during training.

    The algorithm works by:
    1. Computing cosine similarity between encoder and decoder features
    2. Identifying well-reconstructed points (those with high cosine similarity)
    3. Reducing the gradient contribution of these easy points by factor
    4. Focusing training on harder-to-reconstruct points

    This prevents the decoder from learning to reconstruct anomalous patterns, which is
    crucial for effective anomaly detection during inference.

    Args:
        p (float): Percentage of well-reconstructed (easy) points to down-weight.
            Higher values (closer to 1.0) down-weight more points, making training
            focus on fewer, harder examples. Default is 0.9 (down-weight 90% of easy points).
        factor (float): Gradient reduction factor for well-reconstructed points.
            Lower values reduce gradient contribution more aggressively. Default is 0.1
            (reduce gradients to 10% of the original value).

    Note:
        Despite the name "hard mining", this loss actually down-weights easy examples
        rather than up-weighting hard ones. The naming follows the original implementation
        for consistency.
    """

    def __init__(self, p_final: float = 0.9, p_schedule_steps: int = 1000, factor: float = 0.1) -> None:
        """Initialize the CosineHardMiningLoss.

        Args:
            p_final (float): Final percentage of well-reconstructed points to down-weight.
                This is used to clip the p value during training. Default is 0.9.
            p_schedule_steps (int): Number of steps over which to schedule the p value.
                This allows gradual adjustment of the p value during training.After these many steps,
                the p value will be set to p_final. Default is 1000.
            factor (float): Gradient reduction factor for well-reconstructed points (0.0 to 1.0).
                Lower values reduce gradient contribution more aggressively. Default is 0.1.
        """
        super().__init__()

        self.p_final = p_final
        self.factor = factor
        self.p_schedule_steps = p_schedule_steps
        self.p = 0.0  # This is updated before calculating the loss

    def forward(
        self,
        encoder_features: list[torch.Tensor],
        decoder_features: list[torch.Tensor],
        global_step: int,
    ) -> torch.Tensor:
        """Forward pass of the cosine hard mining loss.

        Computes cosine similarity loss between encoder and decoder features while
        applying gradient modification to down-weight well-reconstructed points.

        Args:
            encoder_features: List of feature tensors from encoder layers.
                Each tensor should have a shape (batch_size, num_features, height, width).
            decoder_features: List of corresponding feature tensors from decoder layers.
                Must have the same length and compatible shapes as encoder_features.
            global_step (int): Current training step, used to update the p value schedule.

        Returns:
            Computed loss value averaged across all feature layers.

        Note:
            The encoder features are detached to prevent gradient flow through the encoder,
            focusing training only on the decoder parameters.
        """
        # Update the p value based on the global step
        self._update_p_schedule(global_step)
        cos_loss = torch.nn.CosineSimilarity()
        loss = torch.tensor(0.0, device=encoder_features[0].device)
        for item in range(len(encoder_features)):
            en_ = encoder_features[item].detach()
            de_ = decoder_features[item]
            with torch.no_grad():
                point_dist = 1 - cos_loss(en_, de_).unsqueeze(1)
            k = max(1, int(point_dist.numel() * (1 - self.p)))
            thresh = torch.topk(point_dist.reshape(-1), k=k)[0][-1]

            loss += torch.mean(1 - cos_loss(en_.reshape(en_.shape[0], -1), de_.reshape(de_.shape[0], -1)))

            partial_func = partial(
                self._modify_grad,
                indices_to_modify=point_dist < thresh,
                gradient_multiply_factor=self.factor,
            )
            de_.register_hook(partial_func)

        return loss / len(encoder_features)

    @staticmethod
    def _modify_grad(
        x: torch.Tensor,
        indices_to_modify: torch.Tensor,
        gradient_multiply_factor: float = 0.0,
    ) -> torch.Tensor:
        """Modify gradients based on indices and factor.

        Args:
            x: Input tensor
            indices_to_modify: Boolean indices indicating which elements to modify
            gradient_multiply_factor: Factor to multiply the selected gradients by

        Returns:
            Modified tensor
        """
        indices_to_modify = indices_to_modify.expand_as(x)
        result = x.clone()
        result[indices_to_modify] = result[indices_to_modify] * gradient_multiply_factor
        return result

    def _update_p_schedule(self, global_step: int) -> None:
        """Update the percentage of well-reconstructed points to down-weight based on the global step.

        Args:
            global_step (int): Current training step, used to update the p value schedule.
        """
        self.p = min(self.p_final * global_step / self.p_schedule_steps, self.p_final)
