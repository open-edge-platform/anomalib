# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model for the Dinomaly model implementation.

Based on PyTorch Implementation of "Dinomaly" by guojiajeremy
Reference: https://github.com/guojiajeremy/Dinomaly
License: MIT

See Also:
    :class:`anomalib.models.image.dinomaly.lightning_model.Dinomaly`:
        Dinomaly Lightning model.
"""

import math
from functools import partial

import torch
import torch.nn.functional as F  # noqa: N812
from timm.layers.drop import DropPath
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.components import GaussianBlur2d
from anomalib.models.image.dinomaly.components import CosineHardMiningLoss, DinomalyMLP, LinearAttention
from anomalib.models.image.dinomaly.components import load as load_dinov2_model

# Encoder architecture configurations for DINOv2 models.
# The target layers are the
DINOV2_ARCHITECTURES = {
    "small": {"embed_dim": 384, "num_heads": 6, "target_layers": [2, 3, 4, 5, 6, 7, 8, 9]},
    "base": {"embed_dim": 768, "num_heads": 12, "target_layers": [2, 3, 4, 5, 6, 7, 8, 9]},
    "large": {"embed_dim": 1024, "num_heads": 16, "target_layers": [4, 6, 8, 10, 12, 14, 16, 18]},
}

# Default fusion layer configurations
# Instead of comparing layer to layer between encoder and decoder, dinomaly uses
# layer groups to fuse features from multiple layers.
# By Default, the first 4 layers and the last 4 layers are fused.
# Note that these are the layer indices of the encoder and decoder layers used for feature extraction.
DEFAULT_FUSE_LAYERS = [[0, 1, 2, 3], [4, 5, 6, 7]]

# Default values for inference processing
DEFAULT_RESIZE_SIZE = 256
DEFAULT_GAUSSIAN_KERNEL_SIZE = 5
DEFAULT_GAUSSIAN_SIGMA = 4
DEFAULT_MAX_RATIO = 0.01

# Transformer architecture constants
TRANSFORMER_CONFIG: dict[str, float | bool] = {
    "mlp_ratio": 4.0,
    "layer_norm_eps": 1e-8,
    "qkv_bias": True,
    "attn_drop": 0.0,
}


class DinomalyModel(nn.Module):
    """DinomalyModel: Vision Transformer-based anomaly detection model from Dinomaly.

    This is a Vision Transformer-based anomaly detection model that uses an encoder-bottleneck-decoder
    architecture for feature reconstruction.

    The architecture comprises three main components:
    + An Encoder: A pre-trained Vision Transformer (ViT), by default a ViT-Base/14 based dinov2-reg model which
    extracts universal and discriminative features from input images.
    + Bottleneck: A simple MLP that collects feature representations from the encoder's middle-level layers.
    + Decoder: Composed of Transformer layers (by default 8 layers), it learns to reconstruct the middle-level features.

    Args:
        encoder_name (str): Name of the Vision Transformer encoder to use.
            Supports DINOv2 variants like "dinov2reg_vit_base_14".
            Defaults to "dinov2reg_vit_base_14".
        bottleneck_dropout (float): Dropout rate for the bottleneck MLP layer.
            Defaults to 0.2.
        decoder_depth (int): Number of Vision Transformer decoder layers.
            Defaults to 8.
        target_layers (list[int] | None): List of encoder layer indices to extract features from.
            If None, uses [2, 3, 4, 5, 6, 7, 8, 9] for base models.
            For large models, uses [4, 6, 8, 10, 12, 14, 16, 18].
        fuse_layer_encoder (list[list[int]] | None): Layer groupings for encoder feature fusion.
            If None, uses [[0, 1, 2, 3], [4, 5, 6, 7]].
        fuse_layer_decoder (list[list[int]] | None): Layer groupings for decoder feature fusion.
            If None, uses [[0, 1, 2, 3], [4, 5, 6, 7]].
        remove_class_token (bool): Whether to remove class token from features
            before processing. Defaults to False.

    Example:
        >>> model = DinomalyModel(
        ...     encoder_name="dinov2reg_vit_base_14",
        ...     decoder_depth=8,
        ...     bottleneck_dropout=0.2
        ... )
        >>> features = model(batch)
    """

    def __init__(
        self,
        encoder_name: str = "dinov2reg_vit_base_14",
        bottleneck_dropout: float = 0.2,
        decoder_depth: int = 8,
        target_layers: list[int] | None = None,
        fuse_layer_encoder: list[list[int]] | None = None,
        fuse_layer_decoder: list[list[int]] | None = None,
        remove_class_token: bool = False,
    ) -> None:
        super().__init__()

        if target_layers is None:
            # 8 middle layers of the encoder are used for feature extraction.
            target_layers = [2, 3, 4, 5, 6, 7, 8, 9]

        # Instead of comparing layer to layer between encoder and decoder, dinomaly uses
        # layer groups to fuse features from multiple layers.
        if fuse_layer_encoder is None:
            fuse_layer_encoder = DEFAULT_FUSE_LAYERS
        if fuse_layer_decoder is None:
            fuse_layer_decoder = DEFAULT_FUSE_LAYERS

        encoder = load_dinov2_model(encoder_name)

        # Extract architecture configuration based on the model name
        arch_config = self._get_architecture_config(encoder_name, target_layers)
        embed_dim = arch_config["embed_dim"]
        num_heads = arch_config["num_heads"]
        target_layers = arch_config["target_layers"]

        # Add validation
        if decoder_depth <= 1:
            msg = f"decoder_depth must be greater than 1, got {decoder_depth}"
            raise ValueError(msg)

        bottleneck = []
        bottle_neck_mlp = DinomalyMLP(
            in_features=embed_dim,
            hidden_features=embed_dim * 4,
            out_features=embed_dim,
            act_layer=nn.GELU,
            drop=bottleneck_dropout,
            bias=False,
            apply_input_dropout=True,  # Apply dropout to input
        )
        bottleneck.append(bottle_neck_mlp)
        bottleneck = nn.ModuleList(bottleneck)

        decoder = []
        for _ in range(decoder_depth):
            # Extract and validate config values for type safety
            mlp_ratio_val = TRANSFORMER_CONFIG["mlp_ratio"]
            assert isinstance(mlp_ratio_val, float)
            qkv_bias_val = TRANSFORMER_CONFIG["qkv_bias"]
            assert isinstance(qkv_bias_val, bool)
            layer_norm_eps_val = TRANSFORMER_CONFIG["layer_norm_eps"]
            assert isinstance(layer_norm_eps_val, float)
            attn_drop_val = TRANSFORMER_CONFIG["attn_drop"]
            assert isinstance(attn_drop_val, float)

            decoder_block = DecoderViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio_val,
                qkv_bias=qkv_bias_val,
                norm_layer=partial(nn.LayerNorm, eps=layer_norm_eps_val),  # type: ignore[arg-type]
                attn_drop=attn_drop_val,
                attn=LinearAttention,
            )
            decoder.append(decoder_block)
        decoder = nn.ModuleList(decoder)

        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token

        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0

        # Initialize Gaussian blur for anomaly map smoothing
        self.gaussian_blur = GaussianBlur2d(
            sigma=DEFAULT_GAUSSIAN_SIGMA,
            channels=1,
            kernel_size=DEFAULT_GAUSSIAN_KERNEL_SIZE,
        )

        self.loss_fn = CosineHardMiningLoss()

    def get_encoder_decoder_outputs(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Extract and process features through encoder and decoder.

        This method processes input images through the DINOv2 encoder to extract
        features from target layers, fuses them through a bottleneck MLP, and
        reconstructs them using the decoder. Features are reshaped for spatial
        anomaly map computation.

        Args:
            x (torch.Tensor): Input images with shape (B, C, H, W).

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: Tuple containing:
                - en: List of fused encoder features reshaped to spatial dimensions
                - de: List of fused decoder features reshaped to spatial dimensions
        """
        x = self.encoder.prepare_tokens(x)

        encoder_features = []
        decoder_features = []

        for i, block in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = block(x)
            else:
                continue
            if i in self.target_layers:
                encoder_features.append(x)
        side = int(math.sqrt(encoder_features[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            encoder_features = [e[:, 1 + self.encoder.num_register_tokens :, :] for e in encoder_features]

        x = self._fuse_feature(encoder_features)
        for _i, block in enumerate(self.bottleneck):
            x = block(x)

        # attn_mask is explicitly set to None to disable attention masking.
        # This will not have any effect as it was essentially set to None in the original implementation
        # as well but was configurable to be not None for testing, if required.
        for _i, block in enumerate(self.decoder):
            x = block(x, attn_mask=None)
            decoder_features.append(x)
        decoder_features = decoder_features[::-1]

        en = [self._fuse_feature([encoder_features[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self._fuse_feature([decoder_features[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        # Process features for spatial output
        en = self._process_features_for_spatial_output(en, side)
        de = self._process_features_for_spatial_output(de, side)
        return en, de

    def forward(self, batch: torch.Tensor, global_step: int | None = None) -> torch.Tensor | InferenceBatch:
        """Forward pass of the Dinomaly model.

        During training, the model extracts features from the encoder and decoder
        and returns them for loss computation. During inference, it computes
        anomaly maps by comparing encoder and decoder features using cosine similarity,
        applies Gaussian smoothing, and returns anomaly scores and maps.

        Args:
            batch (torch.Tensor): Input batch of images with shape (B, C, H, W).
            global_step (int | None): Current training step, used for loss computation.

        Returns:
            torch.Tensor | InferenceBatch:
                - During training: Dictionary containing encoder and decoder features
                  for loss computation.
                - During inference: InferenceBatch with pred_score (anomaly scores)
                  and anomaly_map (pixel-level anomaly maps).

        """
        en, de = self.get_encoder_decoder_outputs(batch)
        image_size = batch.shape[2]

        if self.training:
            if global_step is None:
                error_msg = "global_step must be provided during training"
                raise ValueError(error_msg)

            return self.loss_fn(encoder_features=en, decoder_features=de, global_step=global_step)

        # If inference, calculate anomaly maps, predictions, from the encoder and decoder features.
        anomaly_map, _ = self.calculate_anomaly_maps(en, de, out_size=image_size)
        anomaly_map_resized = anomaly_map.clone()

        # Resize anomaly map for processing
        if DEFAULT_RESIZE_SIZE is not None:
            anomaly_map = F.interpolate(anomaly_map, size=DEFAULT_RESIZE_SIZE, mode="bilinear", align_corners=False)

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

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map_resized)

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

    def _process_features_for_spatial_output(
        self,
        features: list[torch.Tensor],
        side: int,
    ) -> list[torch.Tensor]:
        """Process features for spatial output by removing tokens and reshaping.

        Args:
            features: List of feature tensors
            side: Side length for spatial reshaping

        Returns:
            List of processed feature tensors with spatial dimensions
        """
        # Remove class token and register tokens if not already removed
        if not self.remove_class_token:
            features = [f[:, 1 + self.encoder.num_register_tokens :, :] for f in features]

        # Reshape to spatial dimensions
        batch_size = features[0].shape[0]
        return [f.permute(0, 2, 1).reshape([batch_size, -1, side, side]).contiguous() for f in features]


class DecoderViTBlock(nn.Module):
    """Vision Transformer decoder block with attention and MLP layers."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float | None = None,
        qkv_bias: bool | None = None,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        attn: type[nn.Module] = LinearAttention,
    ) -> None:
        super().__init__()

        # Use default values from TRANSFORMER_CONFIG if not provided
        mlp_ratio_config = TRANSFORMER_CONFIG["mlp_ratio"]
        assert isinstance(mlp_ratio_config, float)
        mlp_ratio = mlp_ratio if mlp_ratio is not None else mlp_ratio_config

        qkv_bias_config = TRANSFORMER_CONFIG["qkv_bias"]
        assert isinstance(qkv_bias_config, bool)
        qkv_bias = qkv_bias if qkv_bias is not None else qkv_bias_config

        attn_drop_config = TRANSFORMER_CONFIG["attn_drop"]
        assert isinstance(attn_drop_config, float)
        attn_drop = attn_drop if attn_drop is not None else attn_drop_config

        self.norm1 = norm_layer(dim)
        self.attn = attn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DinomalyMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
            apply_input_dropout=False,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through decoder block."""
        if attn_mask is not None:
            y, attn = self.attn(self.norm1(x), attn_mask=attn_mask)
        else:
            y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        return x
