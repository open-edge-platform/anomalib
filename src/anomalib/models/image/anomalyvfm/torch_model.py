# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model for the AnomalyVFM model implementation.

See Also:
    :class:`anomalib.models.image.anomalyvfm.lightning_model.AnomalyVFM`:
        AnomalyVFM Lightning model.
"""

try:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    _HAS_HF_DEPS = True
except ImportError:
    _HAS_HF_DEPS = False

import math

import torch
from torch import nn
from torch.nn import functional

from anomalib import PrecisionType

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
        self.decoder = SimpleDecoder(feat_dim, 1, 1)
        self.predictor = SimplePredictor(feat_dim * 3)

        if not _HAS_HF_DEPS:
            msg = "Please install them using: pip install anomalib[huggingface]"
            raise ImportError(msg)
        weights_path = hf_hub_download(
            repo_id="MaticFuc/anomalyvfm_radio",
            filename="model.safetensors",
            revision="17654e763c8fae5ae1c44e2ec421a427783d6196",
            local_files_only=True,
        )
        safe_state_dict = load_file(weights_path)
        self.load_state_dict(safe_state_dict)

        self.mean_kernel = nn.AvgPool2d((5, 5), 1, 5 // 2)

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to compute anomaly scores and masks.

        Args:
            img (torch.Tensor): Input image batch of shape (b, C, H, W).

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): A tuple containing:
                - anomaly_score (torch.Tensor): Image-level anomaly predictions.
                - anomaly_mask (torch.Tensor): Pixel-level anomaly prediction masks.
        """
        b = img.shape[0]

        device_type = img.device.type
        dtype = torch.float32 if self.precision is None or self.precision == PrecisionType.FLOAT32 else torch.bfloat16

        with torch.autocast(device_type=device_type, dtype=dtype), torch.no_grad():
            summary, ftrs = self.model(img)
            feat_size = int(math.sqrt(img.shape[1]))
            ftrs = ftrs.permute(0, 2, 1)
            ftrs = ftrs.reshape(b, -1, feat_size, feat_size)

            anomaly_score = self.predictor(summary).sigmoid()
            anomaly_maps, _ = self.decoder(ftrs)
            anomaly_maps = anomaly_maps.sigmoid()
            anomaly_maps = self.mean_kernel(anomaly_maps)
            anomaly_maps = functional.interpolate(
                anomaly_maps,
                size=feat_size * self.model.patch_size,
                mode="bilinear",
                align_corners=False,
            )

        return anomaly_score, anomaly_maps


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
