# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model for the AnomalyVFM model implementation.

See Also:
    :class:`anomalib.models.image.anomalyvfm.lightning_model.AnomalyVFM`:
        AnomalyVFM Lightning model.
"""

import torch
from torch import nn
from torchvision.transforms import v2

from .components.decoder import SimpleDecoder, SimplePredictor
from .components.dora import add_peft
from .components.radio import RADIOModel


class AnomalyVFMModel(
    nn.Module,
):
    """AnomalyVFM PyTorch model.

    This model integrates a base Vision Foundation Model (RADIO) configured with 
    Parameter-Efficient Fine-Tuning (PEFT), alongside a simple decoder for generating 
    pixel-level anomaly masks and a simple predictor for image-level anomaly scores.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = BaseModel()
        self.model.add_peft()
        feat_dim = self.model.feature_dim
        self.feat_size = self.model.H // self.model.patch_size
        self.decoder = SimpleDecoder(feat_dim, 1, 1)
        self.predictor = SimplePredictor(feat_dim * 3)

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to compute anomaly scores and masks.

        Args:
            img (torch.Tensor): Input image batch of shape (B, C, H, W).

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): A tuple containing:
                - anomaly_score (torch.Tensor): Image-level anomaly predictions.
                - anomaly_mask (torch.Tensor): Pixel-level anomaly prediction masks.
        """
        B = img.shape[0]

        device_type = img.device.type

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16), torch.no_grad():
            summary, ftrs = self.model(img)
            ftrs = ftrs.permute(0, 2, 1)
            ftrs = ftrs.reshape(B, -1, self.feat_size, self.feat_size)

            anomaly_score = self.predictor(summary).sigmoid()
            anomaly_mask, _ = self.decoder(ftrs)
            anomaly_mask = anomaly_mask.sigmoid()

        return anomaly_score.float(), anomaly_mask.float()


class BaseModel(nn.Module):
    """Base model wrapper for the RADIO vision foundation model.
    
    Initializes the RADIO model backbone and defines default image dimensions 
    and patch sizes used for spatial feature extraction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = RADIOModel()
        self.feature_dim = 1024
        self.patch_size = 16
        self.H = 768

    def get_img_transform(self) -> v2.Compose:
        """Get the default image transformation pipeline.

        Returns:
            (v2.Compose): Torchvision v2 transforms for resizing and tensor conversion.
        """
        return v2.Compose(
            [
                v2.Resize((self.H, self.H)),
                v2.ToTensor(),
            ],
        )

    def add_peft(self, r: int = 64) -> None:
        """Add Parameter-Efficient Fine-Tuning (PEFT) adaptors to the network.

        Args:
            r (int): The rank for the DoRA/LoRA adapter layers. Default is 64.
        """
        add_peft(self.net, r=r)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the base network.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): A tuple containing:
                - summary (torch.Tensor): Extracted summary features.
                - spatial_features (torch.Tensor): Extracted spatial patch features.
        """
        output = self.net(x)
        return output[0], output[1]