# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Feature extractor using timm models and any nn.Module torch model.

This module provides a feature extractor implementation that leverages the timm
library to extract intermediate features from various CNN architectures. If a
nn.Module is passed as backbone argument, the TorchFX feature extractor is
used to extract features of the given layers.

Example:
    >>> import torch
    >>> from anomalib.models.components.feature_extractors import (
    ...     TimmFeatureExtractor
    ... )
    >>> # Initialize feature extractor
    >>> extractor = TimmFeatureExtractor(
    ...     backbone="resnet18",
    ...     layers=["layer1", "layer2", "layer3"]
    ... )
    >>> # Extract features from input
    >>> inputs = torch.randn(32, 3, 256, 256)
    >>> features = extractor(inputs)
    >>> # Access features by layer name
    >>> print(features["layer1"].shape)
    torch.Size([32, 64, 64, 64])
"""

import logging
import math
from collections.abc import Sequence

import timm
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torchvision.models.feature_extraction import create_feature_extractor

from .utils import dryrun_find_featuremap_dims

logger = logging.getLogger(__name__)

# DINOv2 positional-embedding interpolation offset. The original DINOv2/Facebook
# implementation interpolates the patch position embeddings with a +0.1 coordinate
# offset, mode="bicubic" and antialias=False. We therefore patch the timm ViT to
# reproduce the original DINOv2 interpolation exactly (timm's default differs and
# degrades downscaled multi-scale features).
_DINOV2_POS_EMBED_OFFSET = 0.1


def _dinov2_resample_pos_embed(
    pos_embed: torch.Tensor,
    new_grid: tuple[int, int],
    num_prefix_tokens: int,
) -> torch.Tensor:
    """Interpolate ViT patch position embeddings facebook-DINOv2-style (offset, no antialias)."""
    h, w = new_grid
    if num_prefix_tokens:
        prefix = pos_embed[:, :num_prefix_tokens]
        patch_pos = pos_embed[:, num_prefix_tokens:]
    else:
        prefix = None
        patch_pos = pos_embed
    dim = patch_pos.shape[-1]
    m = math.isqrt(patch_pos.shape[1])
    sx = (h + _DINOV2_POS_EMBED_OFFSET) / m
    sy = (w + _DINOV2_POS_EMBED_OFFSET) / m
    patch_pos = F.interpolate(
        patch_pos.float().reshape(1, m, m, dim).permute(0, 3, 1, 2),
        scale_factor=(sx, sy),
        mode="bicubic",
        antialias=False,
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim).to(pos_embed.dtype)
    return patch_pos if prefix is None else torch.cat([prefix, patch_pos], dim=1)


class TimmFeatureExtractor(nn.Module):
    """Extract intermediate features from timm models or any nn.Module torch model.

    Two extraction modes are supported via ``output_fmt``:

    - ``"NCHW"`` (default): uses timm's ``features_only`` API and returns spatial
      feature maps of shape ``(B, C, H, W)``. This is the original behaviour and is
      used by CNN backbones.
    - ``"NLC"``: uses timm's ``forward_intermediates`` API and returns token
      sequences of shape ``(B, N, D)``. This is required for transformer backbones
      (e.g. DINOv2 ViTs) where downstream models operate on tokens. Optionally the
      class/register (prefix) tokens can be prepended and the backbone's final norm
      applied.

    Args:
        backbone (str | nn.Module): Name of the timm model architecture or any torch model to use as backbone.
        layers (Sequence[str | int]): Names of layers from which to extract features. In
            ``"NLC"`` mode these are transformer block names such as ``"blocks.2"``. In
            ``"NCHW"`` mode (string backbone) these may instead be integer indices, which
            are passed directly as ``out_indices`` to timm's ``features_only`` API.
        pre_trained (bool, optional): Whether to use pre-trained weights.
            Defaults to ``True``.
        requires_grad (bool, optional): Whether to compute gradients for the
            backbone. Required for training models like STFPM. Defaults to
            ``False``.
        output_fmt (str, optional): Feature output format, either ``"NCHW"`` (spatial
            maps via ``features_only``) or ``"NLC"`` (token sequences via
            ``forward_intermediates``). Defaults to ``"NCHW"``.
        return_class_token (bool, optional): ``"NLC"`` mode only. If ``True``, the
            prefix tokens (class token followed by register tokens) are prepended to
            the patch tokens, yielding the full ``[CLS, reg..., patches]`` sequence.
            Defaults to ``False``.
        norm (bool, optional): ``"NLC"`` mode only. If ``True``, the backbone's final
            normalization layer is applied to each returned intermediate. Defaults to
            ``False``.
        dynamic_img_size (bool, optional): Passed to ``timm.create_model``. Allows ViT
            backbones to accept input sizes other than their default ``img_size`` by
            interpolating positional embeddings. Defaults to ``False``.
        dinov2_pos_embed (bool, optional): ``"NLC"`` mode only. If ``True``, the ViT's
            ``_pos_embed`` is replaced with the original facebook-DINOv2 positional
            embedding interpolation (``+0.1`` offset, bicubic, no antialias), which timm
            does not reproduce by default. Defaults to ``False``.

    Attributes:
        backbone (str | nn.Module): Name of the backbone model or actual torch backbone model.
        layers (list[str]): Layer names for feature extraction.
        idx (list[int]): Indices mapping layer names to model outputs.
        requires_grad (bool): Whether gradients are computed.
        feature_extractor (nn.Module): The underlying timm model.
        out_dims (list[int]): Output dimensions for each extracted layer.
        reductions (list[int]): ``"NCHW"`` mode only. Reduction stride of each extracted
            feature map relative to the input resolution.
        patch_size (int): ``"NLC"`` mode only. Patch size of the ViT backbone.
        num_prefix_tokens (int): ``"NLC"`` mode only. Number of prefix tokens
            (class token + register tokens).
        num_register_tokens (int): ``"NLC"`` mode only. Number of register tokens.

    Example:
        >>> import torch
        >>> import torchvision
        >>> from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights

        >>> from anomalib.models.components.feature_extractors import (
        ...     TimmFeatureExtractor
        ... )
        >>> # Create extractor
        >>> model = TimmFeatureExtractor(
        ...     backbone="resnet18",
        ...     layers=["layer1", "layer2"]
        ... )
        >>> # Extract features
        >>> inputs = torch.randn(1, 3, 224, 224)
        >>> features = model(inputs)
        >>> # Print shapes
        >>> for name, feat in features.items():
        ...     print(f"{name}: {feat.shape}")
        layer1: torch.Size([1, 64, 56, 56])
        layer2: torch.Size([1, 128, 28, 28])

        >>> # Custom backbone model
        >>> custom_backbone = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        >>> model = TimmFeatureExtractor(
        ...    backbone=custom_backbone,
        ...    layers=["features.6.8"])
        >>> features = model(inputs)
        >>> # Print shapes
        >>> for name, feat in features.items():
        ...     print(f"{name}: {feat.shape}")
        features.6.8: torch.Size([32, 304, 8, 8])

    """

    def __init__(
        self,
        backbone: str | nn.Module,
        layers: Sequence[str | int],
        pre_trained: bool = True,
        requires_grad: bool = False,
        output_fmt: str = "NCHW",
        return_class_token: bool = False,
        norm: bool = False,
        dynamic_img_size: bool = False,
        dinov2_pos_embed: bool = False,
    ) -> None:
        super().__init__()

        if output_fmt not in {"NCHW", "NLC"}:
            msg = f"output_fmt must be one of 'NCHW' or 'NLC', got '{output_fmt}'."
            raise ValueError(msg)

        if dinov2_pos_embed and output_fmt != "NLC":
            msg = "dinov2_pos_embed is only supported in 'NLC' mode."
            raise ValueError(msg)

        self.backbone = backbone
        self.layers = list(layers)
        self.requires_grad = requires_grad
        self.output_fmt = output_fmt
        self.return_class_token = return_class_token
        self.norm = norm
        self.dinov2_pos_embed = dinov2_pos_embed

        if isinstance(backbone, nn.Module):
            self.feature_extractor = create_feature_extractor(
                backbone,
                return_nodes={layer: layer for layer in self.layers},
            )
            layer_metadata = dryrun_find_featuremap_dims(self.feature_extractor, (256, 256), layers=self.layers)
            self.out_dims = [feature_info["num_features"] for layer_name, feature_info in layer_metadata.items()]

        elif isinstance(backbone, str) and output_fmt == "NLC":
            # Token mode: use the full model with forward_intermediates (e.g. ViT/DINOv2).
            self.idx = [self._block_name_to_idx(layer) for layer in self.layers]
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=pre_trained,
                pretrained_cfg=None,
                dynamic_img_size=dynamic_img_size,
            )
            self.patch_size = self.feature_extractor.patch_embed.patch_size[0]
            self.num_prefix_tokens = getattr(self.feature_extractor, "num_prefix_tokens", 1)
            self.num_register_tokens = self.num_prefix_tokens - 1
            embed_dim = self.feature_extractor.num_features
            self.out_dims = [embed_dim] * len(self.layers)
            if self.dinov2_pos_embed:
                # Reproduce the original facebook-DINOv2 positional-embedding interpolation.
                self.feature_extractor._pos_embed = self._dinov2_pos_embed  # noqa: SLF001

        elif isinstance(backbone, str):
            # Integer layers are treated as out_indices for timm's features_only API;
            # otherwise the layer names are mapped to their indices.
            indices_given = all(isinstance(layer, int) for layer in self.layers)
            self.idx = list(self.layers) if indices_given else self._map_layer_to_idx()
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=pre_trained,
                pretrained_cfg=None,
                features_only=True,
                exportable=True,
                out_indices=self.idx,
            )
            if indices_given:
                # Resolve indices to the actual module names so outputs are keyed by name.
                info = self.feature_extractor.feature_info.info
                self.layers = [info[i]["module"] for i in self.idx]
            self.out_dims = self.feature_extractor.feature_info.channels()
            self.reductions = self.feature_extractor.feature_info.reduction()

        else:
            msg = f"Backbone of type {type(backbone)} must be of type str or nn.Module."
            raise TypeError(msg)

        self._features = {layer: torch.empty(0) for layer in self.layers}

    @staticmethod
    def _block_name_to_idx(layer: str) -> int:
        """Parse a transformer block layer name such as ``"blocks.2"`` into its index.

        Args:
            layer (str): Block layer name of the form ``"blocks.<int>"``.

        Returns:
            int: The integer block index.
        """
        try:
            return int(layer.rsplit(".", 1)[-1])
        except ValueError as exc:
            msg = f"In 'NLC' mode, layer names must be of the form 'blocks.<int>', got '{layer}'."
            raise ValueError(msg) from exc

    def _dinov2_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Replacement for timm ViT ``_pos_embed`` using the original facebook-DINOv2 interpolation.

        Args:
            x (torch.Tensor): Patch-embedded input of shape ``(B, H, W, C)``.

        Returns:
            torch.Tensor: Token sequence with positional (and prefix) embeddings added.
        """
        model = self.feature_extractor
        to_cat = []
        if model.cls_token is not None:
            to_cat.append(model.cls_token.expand(x.shape[0], -1, -1))
        if model.reg_token is not None:
            to_cat.append(model.reg_token.expand(x.shape[0], -1, -1))

        b, h, w, c = x.shape
        pos_embed = _dinov2_resample_pos_embed(
            model.pos_embed,
            (h, w),
            num_prefix_tokens=0 if model.no_embed_class else model.num_prefix_tokens,
        )
        x = x.view(b, -1, c)
        if model.no_embed_class:
            x = x + pos_embed
            if to_cat:
                x = torch.cat([*to_cat, x], dim=1)
        else:
            if to_cat:
                x = torch.cat([*to_cat, x], dim=1)
            x = x + pos_embed
        return model.pos_drop(x)

    def _map_layer_to_idx(self) -> list[int]:
        """Map layer names to their indices in the model's output.

        Returns:
            list[int]: Indices corresponding to the requested layer names.

        Note:
            If a requested layer is not found in the model, it is removed from
            ``self.layers`` and a warning is logged.
        """
        idx = []
        model = timm.create_model(
            self.backbone,
            pretrained=False,
            features_only=True,
            exportable=True,
        )
        # model.feature_info.info returns list of dicts containing info,
        # inside which "module" contains layer name
        layer_names = [info["module"] for info in model.feature_info.info]
        for layer in self.layers:
            try:
                idx.append(layer_names.index(layer))
            except ValueError:  # noqa: PERF203
                msg = f"Layer {layer} not found in model {self.backbone}. Available layers: {layer_names}"
                logger.warning(msg)
                # Remove unfound key from layer dict
                self.layers.remove(layer)

        return idx

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from the input tensor.

        Args:
            inputs (torch.Tensor): Input tensor of shape
                ``(batch_size, channels, height, width)``.

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping layer names to their
            feature tensors.

        Example:
            >>> import torch
            >>> from anomalib.models.components.feature_extractors import (
            ...     TimmFeatureExtractor
            ... )
            >>> model = TimmFeatureExtractor(
            ...     backbone="resnet18",
            ...     layers=["layer1"]
            ... )
            >>> inputs = torch.randn(1, 3, 224, 224)
            >>> features = model(inputs)
            >>> features["layer1"].shape
            torch.Size([1, 64, 56, 56])
        """
        if self.output_fmt == "NLC":
            return self._forward_nlc(inputs)

        if self.requires_grad:
            features = self.feature_extractor(inputs)
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                features = self.feature_extractor(inputs)
        if not isinstance(features, dict):
            features = dict(zip(self.layers, features, strict=True))
        return features

    def _forward_nlc(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract token-sequence features via timm's ``forward_intermediates``.

        Args:
            inputs (torch.Tensor): Input tensor of shape ``(B, C, H, W)``.

        Returns:
            dict[str, torch.Tensor]: Mapping of block layer names to token tensors.
                Each tensor has shape ``(B, N, D)`` (patch tokens), or
                ``(B, P + N, D)`` with the ``[CLS, reg..., patches]`` prefix tokens
                prepended when ``return_class_token`` is ``True``.
        """
        if self.requires_grad:
            intermediates = self._run_intermediates(inputs)
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                intermediates = self._run_intermediates(inputs)

        features = {}
        for layer, out in zip(self.layers, intermediates, strict=True):
            if self.return_class_token:
                patch_tokens, prefix_tokens = out
                features[layer] = torch.cat([prefix_tokens, patch_tokens], dim=1)
            else:
                features[layer] = out
        return features

    def _run_intermediates(self, inputs: torch.Tensor) -> list:
        """Call the backbone's ``forward_intermediates`` with the configured options."""
        return self.feature_extractor.forward_intermediates(
            inputs,
            indices=self.idx,
            norm=self.norm,
            return_prefix_tokens=self.return_class_token,
            output_fmt="NLC",
            intermediates_only=True,
        )
