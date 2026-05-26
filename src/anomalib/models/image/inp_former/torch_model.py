# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model for the INP-Former model implementation.

Based on PyTorch Implementation of "INP-Former" by luow23
Reference: https://github.com/luow23/INP-Former
License: MIT

See Also:
    :class:`anomalib.models.image.inp_former.lightning_model.InpFormer`:
        INP-Former Lightning model.
"""

import math
from functools import partial

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.components import GaussianBlur2d
from anomalib.models.components.dinov2 import DinoV2Loader
from anomalib.models.image.dinomaly.components import DinomalyMLP
from anomalib.models.image.dinomaly.components import vision_transformer as dinomaly_vision_transformer
from anomalib.models.image.inp_former.components.layers import AggregationBlock, PrototypeBlock
from anomalib.models.image.inp_former.components.loss import GlobalCosineHmAdaptiveLoss

DEFAULT_FUSE_LAYERS = [[0, 1, 2, 3], [4, 5, 6, 7]]


# Encoder architecture configurations for DINOv2 models.
# The target layers are the
DINOV2_ARCHITECTURES = {
    "small": {"embed_dim": 384, "num_heads": 6, "target_layers": [2, 3, 4, 5, 6, 7, 8, 9]},
    "base": {"embed_dim": 768, "num_heads": 12, "target_layers": [2, 3, 4, 5, 6, 7, 8, 9]},
    "large": {"embed_dim": 1024, "num_heads": 16, "target_layers": [4, 6, 8, 10, 12, 14, 16, 18]},
}

# Default values for inference processing
DEFAULT_RESIZE_SIZE = 256
DEFAULT_GAUSSIAN_KERNEL_SIZE = 5
DEFAULT_GAUSSIAN_SIGMA = 4
DEFAULT_MAX_RATIO = 0.01


class InpFormerModel(nn.Module):
    """PyTorch module implementing the INP-Former anomaly detection model.

    The model consists of four components: a frozen pre-trained Vision Transformer
    encoder, an INP Extractor that aggregates encoder patch tokens into a small set of
    Intrinsic Normal Prototypes (INPs) via cross-attention with learnable queries, a
    bottleneck that fuses multi-scale encoder features, and an INP-Guided Decoder that
    reconstructs normal features using the INPs as keys and values. Anomaly scores are
    computed from the per-token cosine discrepancy between encoder and decoder features
    at multiple scales.

    Args:
        encoder_name (str): Name of the pre-trained Vision Transformer backbone to use
            as the encoder (e.g., a DINOv2 variant).
        inp_num (int): Number of Intrinsic Normal Prototypes to extract per image.
            Defaults to ``6``.
        target_layers (list[int] | None): Indices of encoder layers from which to
            extract intermediate features for reconstruction. Defaults to ``None``.
        fuse_layer_encoder (list[list[int]] | None): Groups of encoder layer indices to
            fuse together when forming the multi-scale encoder feature targets.
            Defaults to ``None``.
        fuse_layer_decoder (list[list[int]] | None): Groups of decoder layer indices to
            fuse together when forming the multi-scale decoder outputs.
            Defaults to ``None``.
        remove_class_token (bool): If ``True``, the class token is dropped from the
            patch token sequence before INP extraction and reconstruction.
            Defaults to ``False``.
        encoder_require_grad_layer (list[int]): Indices of encoder layers that should
            remain trainable; all other encoder layers are frozen. Defaults to ``[]``
            (fully frozen encoder).
    """

    def __init__(
        self,
        encoder_name: str,
        inp_num: int = 6,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = False,
        encoder_require_grad_layer: list[int] | None = None,
    ) -> None:
        super().__init__()
        if encoder_require_grad_layer is None:
            self.encoder_require_grad_layer: list[int] = []
        self.remove_class_token = remove_class_token

        if target_layers is None:
            self.target_layers = [2, 3, 4, 5, 6, 7, 8, 9]

        if fuse_layer_encoder is None:
            self.fuse_layer_encoder = DEFAULT_FUSE_LAYERS
        if fuse_layer_decoder is None:
            self.fuse_layer_decoder = DEFAULT_FUSE_LAYERS

        self.encoder = DinoV2Loader(vit_factory=dinomaly_vision_transformer).load(encoder_name)

        # Extract architecture configuration based on the model name
        arch_config = self._get_architecture_config(encoder_name, target_layers)
        embed_dim = arch_config["embed_dim"]
        num_heads = arch_config["num_heads"]
        target_layers = arch_config["target_layers"]

        # INP
        self.prototype_token = nn.ParameterList([nn.Parameter(torch.randn(inp_num, embed_dim)) for _ in range(1)])

        # Bottleneck MLP for feature fusion
        bottleneck = []
        bottle_neck_mlp = DinomalyMLP(
            in_features=embed_dim,
            hidden_features=embed_dim * 4,
            out_features=embed_dim,
            act_layer=nn.GELU,
            drop=0.0,
            bias=True,
            apply_input_dropout=False,
        )
        bottleneck.append(bottle_neck_mlp)
        self.bottleneck = nn.ModuleList(bottleneck)

        # INP Extractor
        inp_extractor = []
        for _ in range(1):
            blk = AggregationBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
            )
            inp_extractor.append(blk)
        self.aggregation = nn.ModuleList(inp_extractor)

        # INP Decoder
        inp_guided_decoder = []
        for _ in range(8):
            blk = PrototypeBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-8),
            )
            inp_guided_decoder.append(blk)
        self.decoder = nn.ModuleList(inp_guided_decoder)

        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0

        # Initialize Gaussian blur for anomaly map smoothing
        self.gaussian_blur = GaussianBlur2d(
            sigma=DEFAULT_GAUSSIAN_SIGMA,
            channels=1,
            kernel_size=DEFAULT_GAUSSIAN_KERNEL_SIZE,
        )

        self.loss = GlobalCosineHmAdaptiveLoss()

    def get_inp_loss(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """INP coherence loss helps to ensure that INPs represent normal features.

        It minimizes the distances between individual normal features and the corresponding
        nearest INP.

        Args:
            query (torch.Tensor): Fused encoder features (element-wise average).
            keys (torch.Tensor): Prototype visual token.

        Returns:
            torch.Tensor: INP coherence loss.

        """
        self.distribution = 1.0 - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        self.distance, self.cluster_index = torch.min(self.distribution, dim=2)
        return self.distance.mean()

    def forward(self, batch: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Forward pass of the INPFormerModel model.

        During training, the model extracts features from the encoder and decoder
        and returns them for loss computation. During inference, it computes
        anomaly maps by comparing encoder and decoder features using cosine similarity,
        applies Gaussian smoothing, and returns anomaly scores and maps.

        Args:
            batch (torch.Tensor): Input batch of images with shape (B, C, H, W).

        Returns:
            torch.Tensor | InferenceBatch:
                - During training: Encoder and decoder features, INP coherence loss.
                - During inference: InferenceBatch with pred_score (anomaly scores)
                  and anomaly_map (pixel-level anomaly maps).

        """
        en, de, inp_loss = self.get_encoder_decoder_inploss(batch)
        image_size = (batch.shape[2], batch.shape[3])

        if self.training:
            return self.loss(encoder_features=en, decoder_features=de, inp_loss=inp_loss)

        # If inference, calculate anomaly maps, predictions, from the encoder and decoder features.
        anomaly_map, _ = self.calculate_anomaly_maps(en, de, out_size=image_size)

        # Apply Gaussian smoothing
        anomaly_map = self.gaussian_blur(anomaly_map)

        # Calculate anomaly score
        if DEFAULT_MAX_RATIO == 0:
            sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
        else:
            anomaly_map_flat = anomaly_map.flatten(1)
            sp_score = torch.sort(anomaly_map_flat, dim=1, descending=True)[0][
                :,
                : int(anomaly_map_flat.shape[1] * DEFAULT_MAX_RATIO),
            ]
            sp_score = sp_score.mean(dim=1)
        pred_score = sp_score

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    @staticmethod
    def calculate_anomaly_maps(
        source_feature_maps: list[torch.Tensor],
        target_feature_maps: list[torch.Tensor],
        out_size: int | tuple[int, int] = 392,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Calculate anomaly maps by comparing encoder and decoder features.

        Computes pixel-level anomaly maps by calculating cosine similarity between
        corresponding encoder (source) and decoder (target) feature maps. Lower
        cosine similarity indicates a higher anomaly likelihood.

        Args:
            source_feature_maps (list[torch.Tensor]): List of encoder feature maps
                from different layer groups.
            target_feature_maps (list[torch.Tensor]): List of decoder feature maps
                from different layer groups.
            out_size (int | tuple[int, int]): Output size for anomaly maps.
                Defaults to 392.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]: Tuple containing:
                - anomaly_map: Combined anomaly map averaged across all feature scales
                - anomaly_map_list: List of individual anomaly maps for each feature scale
        """
        if not isinstance(out_size, tuple):
            out_size = (out_size, out_size)

        anomaly_map_list = []
        for i in range(len(target_feature_maps)):
            fs = source_feature_maps[i]
            ft = target_feature_maps[i]
            a_map = 1 - F.cosine_similarity(fs, ft)
            a_map = torch.unsqueeze(a_map, dim=1)
            a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
            anomaly_map_list.append(a_map)
        anomaly_map = torch.cat(anomaly_map_list, dim=1).mean(dim=1, keepdim=True)
        return anomaly_map, anomaly_map_list

    @staticmethod
    def _fuse_feature(feat_list: list[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple feature tensors by averaging.

        Takes a list of feature tensors and computes their element-wise average
        to create a fused representation.

        Args:
            feat_list (list[torch.Tensor]): List of feature tensors to fuse.

        Returns:
            torch.Tensor: Averaged feature tensor.

        """
        return torch.stack(feat_list, dim=1).mean(dim=1)

    @staticmethod
    def _get_architecture_config(encoder_name: str, target_layers: list[int] | None) -> dict:
        """Get architecture configuration based on model name.

        Args:
            encoder_name: Name of the encoder model
            target_layers: Override target layers if provided

        Returns:
            Dictionary containing embed_dim, num_heads, and target_layers
        """
        for arch_name, config in DINOV2_ARCHITECTURES.items():
            if arch_name in encoder_name:
                result = config.copy()
                # Override target_layers if explicitly provided
                if target_layers is not None:
                    result["target_layers"] = target_layers
                return result

        msg = f"Architecture not supported. Encoder name must contain one of {list(DINOV2_ARCHITECTURES.keys())}"
        raise ValueError(msg)

    def get_encoder_decoder_inploss(
        self,
        x: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Extract and process features through encoder and decoder.

        This method processes input images through the DINOv2 encoder to extract
        features from target layers, fuses them through a bottleneck MLP, and
        reconstructs them using the decoder. Features are reshaped for spatial
        anomaly map computation. TODO

        Args:
            x (torch.Tensor): Input images with shape (B, C, H, W).

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: Tuple containing:
                - en: List of fused encoder features reshaped to spatial dimensions
                - de: List of fused decoder features reshaped to spatial dimensions
                - inp_loss: INP coherence loss to guide INP Extractor
        """
        x = self.encoder.prepare_tokens(x)
        batch_size = x.shape[0]
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens :, :] for e in en_list]

        x = self._fuse_feature(en_list)

        agg_prototype = self.prototype_token[0]
        for _, blk in enumerate(self.aggregation):
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((batch_size, 1, 1)), x)
        inp_loss = self.get_inp_loss(x, agg_prototype)

        for _, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for _, blk in enumerate(self.decoder):
            x = blk(x, agg_prototype)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self._fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self._fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:
            en = [e[:, 1 + self.encoder.num_register_tokens :, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens :, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de, inp_loss
