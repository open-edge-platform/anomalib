# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cosine Soft Mining Loss for training INP-Former model.

The code is based on the `global_cosine_hm_adaptive` method in the original implementation.
Reference: https://github.com/luow23/INP-Former/blob/5252579e5f401199643fbd16e030175856386f12/utils.py#L130-L151
"""

import torch
from functools import partil


class GlobalCosineHmAdaptiveLoss(torch.nn.Module):
    """Cosine similarity loss with hard mining and adaptive weighting based on distance ratios. 

    This loss function focuses training on hard-to-reconstruct points, enabling the model to learn 
    a sufficient description of normality quickly without becoming overly effective at reconstructing 
    anomalous points.

    Description:
    1. Compute cosine similarity between encoder and decoder features 
    2. High cosine similarity means that points are easy to reconstruct and vice versa
    3. Calculate adaptive weights by amplifying the difference between easy and hard-to-reconstruct points
    4. Apply adaptive weights to modify hard-to-reconstruct points more than easy ones
    5. Training focuses on hard-to-reconstruct points, while modifying easy-to-reconstruct points as well

    Note: 
        This loss is called "Soft Mining" in the paper, presumably referring to using all points in backpropagation 
        with weights applied to focus on hard-to-reconstruct points, in contrast to the "Hard Mining" which uses only 
        hard-to-reconstruct points in backpropagation, ignoring others. The naming follows the original implementation 
        for consistency, which uses the name `global_cosine_hm_adaptive`.

    """

    def __init__(self, power: int = 3) -> None:
        """Initialize the GlobalCosineHmAdaptiveLoss.

        Args:
            power (int): Power to amplify the differences between easy and hard points to reconstruct.
                Default is 3.
        """
        super().__init__()
        self.power = power

    def forward(
            self, 
            encoder_features: list[torch.Tensor], 
            decoder_features: list[torch.Tensor],
            inp_loss: torch.Tensor
        ) -> torch.Tensor:
        """Forward pass of the cosine adaptive loss plus INP loss.

        Args:
            encoder_features: List of feature tensors from encoder layers.
                Each tensor should have a shape (batch_size, num_features, height, width).
            decoder_features: List of corresponding feature tensors from decoder layers.
                Must have the same length and compatible shapes as encoder_features.
            inp_loss: INP coherence loss to minimize the distances between individual 
                normal features and the corresponding nearest INP.

        Returns:
            Computed loss value averaged across all feature layers with INP loss added.

        Note:
            The encoder features are detached to prevent gradient flow through the encoder,
            focusing training only on the decoder parameters.
        """
        cos_loss = torch.nn.CosineSimilarity()
        loss = torch.tensor(0.0, device=encoder_features[0].device)

        for item in range(len(encoder_features)):
            en_ = encoder_features[item].detach()
            de_ = decoder_features[item]
            with torch.no_grad():
                point_dist = 1 - cos_loss(en_, de_).unsqueeze(1)
            mean_dist = point_dist.mean()
            adaptive_weights = (point_dist/mean_dist)**self.power
            loss += torch.mean(1 - cos_loss(en_.reshape(en_.shape[0], -1),
                                            de_.reshape(de_.shape[0], -1)))
            partial_func = partial(self._modify_grad, adaptive_weights=adaptive_weights)
            de_.register_hook(partial_func)

        loss = loss / len(encoder_features)
        loss = loss + 0.2 * inp_loss

        return loss
    
    @staticmethod
    def _modify_grad(
        x: torch.Tensor,
        adaptive_weights: float = 0.0,
    ) -> torch.Tensor:
        """Modify gradients based on adaptive weights.

        Args:
            x: Input tensor
            adaptive_weights: Adaptive weights to modify hard-to-reconstruct points more than easy ones

        Returns:
            Modified tensor
        """
        adaptive_weights = adaptive_weights.expand_as(x)
        result = x.clone()
        result = result * adaptive_weights
        return result