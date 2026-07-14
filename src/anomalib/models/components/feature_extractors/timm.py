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

import functools
import logging
from collections.abc import Sequence
from typing import cast

import timm
import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

from .utils import dryrun_find_featuremap_dims

logger = logging.getLogger(__name__)


def _disable_pos_embed_antialiasing() -> None:
    """Force timm ViT pos-embed resampling to ``antialias=False``.

    timm's ``VisionTransformer._pos_embed`` uses interpolation with
    ``antialias=True``. This operation isn't supported in ONNX export
    and is patched off. It doesn't affect model performance.
    """
    try:
        import timm.models.vision_transformer as vit
    except ImportError:
        return

    original = getattr(vit, "resample_abs_pos_embed", None)
    if original is None or getattr(original, "_anomalib_no_antialias", False):
        return

    @functools.wraps(original)
    def _no_antialias(*args, **kwargs):  # noqa: ANN202
        kwargs["antialias"] = False
        return original(*args, **kwargs)

    setattr(_no_antialias, "_anomalib_no_antialias", True)  # noqa: B010
    vit.resample_abs_pos_embed = _no_antialias


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
        layers (Sequence[str | int]): Names of layers from which to extract features,
            or integer ``out_indices`` for ``features_only`` timm backbones. In
            ``"NLC"`` mode these are transformer block names such as ``"blocks.2"``.
        pre_trained (bool, optional): Whether to use pre-trained weights.
            Defaults to ``True``.
        requires_grad (bool, optional): Whether to compute gradients for the
            backbone. Required for training models like STFPM. Defaults to
            ``False``.
        output_fmt (str, optional): Feature output format, either ``"NCHW"`` (spatial
            maps) or ``"NLC"`` (token sequences). CNN backbones produce ``"NCHW"`` maps via
            timm's ``features_only`` API. Transformer backbones (name contains ``"vit"``, or
            ``output_fmt="NLC"``) use ``forward_intermediates``, which also reshapes patch
            tokens into ``"NCHW"`` spatial maps when requested (e.g. for PatchCore).
            Defaults to ``"NCHW"``.
        return_class_token (bool, optional): Transformer ``"NLC"`` output only. If ``True``,
            the prefix tokens (class token followed by register tokens) are prepended to the
            patch tokens, yielding the full ``[CLS, reg..., patches]`` sequence.
            Defaults to ``False``.
        norm (bool, optional): Transformer backbones only. If ``True``, the backbone's final
            normalization layer is applied to each returned intermediate, yielding well-scaled
            features for nearest-neighbor search (matches AnomalyDINO). Ignored by CNN
            ``features_only`` backbones. Defaults to ``True``.
        dynamic_img_size (bool, optional): Passed to ``timm.create_model`` for transformer
            backbones. Allows ViT backbones to accept input sizes other than their default
            ``img_size`` by interpolating positional embeddings. Ignored by CNN backbones.
            Defaults to ``True``.

    Attributes:
        backbone (str | nn.Module): Name of the backbone model or actual torch backbone model.
        layers (list[str | int]): Layer names (or out-indices) for feature extraction.
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
        norm: bool = True,
        dynamic_img_size: bool = True,
    ) -> None:
        super().__init__()

        if output_fmt not in {"NCHW", "NLC"}:
            msg = f"output_fmt must be one of 'NCHW' or 'NLC', got '{output_fmt}'."
            raise ValueError(msg)

        self.backbone = backbone
        self.layers: list[str | int] = list(layers)
        self.requires_grad = requires_grad
        self.output_fmt = output_fmt
        self.return_class_token = return_class_token
        self.norm = norm
        self.out_dims: Sequence[int]

        # ViT backbones are extracted via ``forward_intermediates`` (the ``features_only`` API cannot
        # handle dynamic ViT input sizes).
        self._uses_intermediates = isinstance(backbone, str) and (output_fmt == "NLC" or "vit" in backbone.lower())

        if isinstance(backbone, nn.Module):
            # ``nn.Module`` backbones require string node names.
            layer_names = cast("list[str]", self.layers)
            self.feature_extractor = create_feature_extractor(
                backbone,
                return_nodes={layer: layer for layer in layer_names},
            )
            layer_metadata = dryrun_find_featuremap_dims(self.feature_extractor, (256, 256), layers=layer_names)
            self.out_dims = [cast("int", feature_info["num_features"]) for feature_info in layer_metadata.values()]

        elif self._uses_intermediates:
            # Transformer mode: use the full model with forward_intermediates (e.g. ViT/DINOv2).
            self.idx = [self._block_name_to_idx(layer) for layer in self.layers]
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=pre_trained,
                pretrained_cfg=None,
                dynamic_img_size=dynamic_img_size,
            )
            _disable_pos_embed_antialiasing()
            self.patch_size = self.feature_extractor.patch_embed.patch_size[0]
            self.num_prefix_tokens = getattr(self.feature_extractor, "num_prefix_tokens", 1)
            self.num_register_tokens = self.num_prefix_tokens - 1
            embed_dim = self.feature_extractor.num_features
            self.out_dims = [embed_dim] * len(self.layers)

        elif isinstance(backbone, str):
            if all(isinstance(layer, int) for layer in self.layers):
                self.idx = [layer for layer in self.layers if isinstance(layer, int)]
            else:
                self.idx = self._map_layer_to_idx()
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=pre_trained,
                pretrained_cfg=None,
                features_only=True,
                exportable=True,
                out_indices=self.idx,
            )
            self.out_dims = self.feature_extractor.feature_info.channels()
            self.reductions = self.feature_extractor.feature_info.reduction()

        else:
            msg = f"Backbone of type {type(backbone)} must be of type str or nn.Module."
            raise TypeError(msg)

        self._features = {layer: torch.empty(0) for layer in self.layers}

    @staticmethod
    def _block_name_to_idx(layer: str | int) -> int:
        """Parse a transformer block layer name such as ``"blocks.2"`` into its index.

        Args:
            layer (str | int): Block layer name of the form ``"blocks.<int>"``, or the
                integer block index itself.

        Returns:
            int: The integer block index.
        """
        if isinstance(layer, int):
            return layer
        try:
            return int(layer.rsplit(".", 1)[-1])
        except ValueError as exc:
            msg = f"In 'NLC' mode, layer names must be of the form 'blocks.<int>', got '{layer}'."
            raise ValueError(msg) from exc

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

    def forward(self, inputs: torch.Tensor) -> dict[str | int, torch.Tensor]:
        """Extract features from the input tensor.

        Args:
            inputs (torch.Tensor): Input tensor of shape
                ``(batch_size, channels, height, width)``.

        Returns:
            dict[str | int, torch.Tensor]: Dictionary mapping layer names (or
            out-indices) to their feature tensors.

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
        if self._uses_intermediates:
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

    def _forward_nlc(self, inputs: torch.Tensor) -> dict[str | int, torch.Tensor]:
        """Extract transformer features via timm's ``forward_intermediates``.

        Args:
            inputs (torch.Tensor): Input tensor of shape ``(B, C, H, W)``.

        Returns:
            dict[str, torch.Tensor]: Mapping of block layer names to feature tensors.
                With ``output_fmt="NLC"`` each tensor has shape ``(B, N, D)`` (patch
                tokens), or ``(B, P + N, D)`` with the ``[CLS, reg..., patches]`` prefix
                tokens prepended when ``return_class_token`` is ``True``. With
                ``output_fmt="NCHW"`` each tensor is a spatial map ``(B, D, H, W)``.
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
            output_fmt=self.output_fmt,
            intermediates_only=True,
        )
