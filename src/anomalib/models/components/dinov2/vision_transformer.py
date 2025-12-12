# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (C) 2025 Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Loader for DINOv2 Vision Transformer models.

This module provides a PyTorch implementation of the DINOv2 Vision Transformer
architecture. It includes utilities for building transformers of different sizes,
handling positional embeddings, preparing tokens with masking, extracting intermediate
layer outputs, and applying initialization schemes compatible with the timm library.

The module forms the backbone for DINO-based feature extraction used in Dinomaly
and related anomaly detection frameworks.
"""

import math
from collections.abc import Callable, Sequence
from functools import partial

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from anomalib.models.components.dinov2.layers import Block, MemEffAttention, Mlp, PatchEmbed, SwiGLUFFNFused


def named_apply(
    fn: Callable[..., object],
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    """Recursively apply a function to a module and all its children.

    Args:
        fn: Callable applied to each visited module.
        module: Module to traverse.
        name: Base name for hierarchical module naming.
        depth_first: If True, apply the function after visiting children.
        include_root: If True, apply function to the root module itself.

    Returns:
        The modified module.
    """
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=full_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    """Container for sequential execution of transformer blocks.

    This utility groups multiple transformer blocks into a single module list
    to improve processing efficiency, particularly in distributed or chunked
    execution settings.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all blocks in the chunk sequentially.

        Args:
            x: Input tensor.

        Returns:
            Tensor output after sequential block processing.
        """
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    """Vision Transformer backbone used in DINOv2.

    This class implements the complete forward pipeline for DINOv2-style Vision
    Transformers, including patch embedding, positional encoding, transformer blocks,
    register token handling, intermediate-layer extraction, and output normalization.

    The architecture supports:
    - Optional register tokens.
    - Chunked transformer blocks for FSDP or memory-efficient training.
    - Mask tokens for masked feature modeling.
    - Interpolated positional encodings for variable-sized input images.
    - Flexible FFN selection (MLP, SwiGLU, fused SwiGLU, identity).

    Args:
        img_size: Input image resolution.
        patch_size: Patch size for patch embedding.
        in_chans: Number of input channels.
        embed_dim: Embedding dimensionality.
        depth: Number of transformer layers.
        num_heads: Number of self-attention heads.
        mlp_ratio: Expansion ratio in feed-forward layers.
        qkv_bias: Whether to include bias in QKV projections.
        ffn_bias: Whether to include bias in FFN layers.
        proj_bias: Whether to include bias in projection layers.
        drop_path_rate: Stochastic depth rate.
        drop_path_uniform: Whether to apply uniform drop-path.
        init_values: Initial values for layer-scale (None disables).
        embed_layer: Patch embedding layer class.
        act_layer: Activation function.
        block_fn: Transformer block class.
        ffn_layer: Feed-forward layer type or constructor.
        block_chunks: Number of chunks to split block sequence into.
        num_register_tokens: Number of extra learned register tokens.
        interpolate_antialias: Whether to apply antialiasing on position interpolation.
        interpolate_offset: Offset to avoid floating point interpolation artifacts.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        init_values: float | None = None,
        embed_layer: type[nn.Module] = PatchEmbed,
        act_layer: type[nn.Module] = nn.GELU,
        block_fn: Callable[..., nn.Module] = Block,
        ffn_layer: str | Callable[..., nn.Module] = "mlp",
        block_chunks: int = 1,
        num_register_tokens: int = 0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
    ) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features: int = embed_dim
        self.embed_dim: int = embed_dim
        self.num_tokens: int = 1
        self.n_blocks: int = depth
        self.num_heads: int = num_heads
        self.patch_size: int = patch_size
        self.num_register_tokens: int = num_register_tokens
        self.interpolate_antialias: bool = interpolate_antialias
        self.interpolate_offset: float = interpolate_offset

        self.patch_embed: nn.Module = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token: nn.Parameter = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed: nn.Parameter = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim),
        )
        self.register_tokens: nn.Parameter | None = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )
        dpr = [drop_path_rate] * depth if drop_path_uniform else torch.linspace(0, drop_path_rate, depth).tolist()

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer in {"swiglu", "swiglufused"}:
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":

            def f(*args: object, **kwargs: object) -> nn.Identity:  # noqa: ARG001
                return nn.Identity()

            ffn_layer = f
        elif isinstance(ffn_layer, str):
            err_msg = f"Unsupported ffn_layer string value '{ffn_layer}'."
            "Supported values are: 'mlp', 'swiglu', 'swiglufused', 'identity'."
            raise NotImplementedError(
                err_msg,
            )
        # else assume callable

        blocks_list: list[nn.Module] = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]

        if block_chunks > 0:
            self.chunked_blocks: bool = True
            chunksize = depth // block_chunks
            chunked_blocks: list[list[nn.Module]] = [
                [nn.Identity()] * i + blocks_list[i : i + chunksize] for i in range(0, depth, chunksize)
            ]
            self.blocks: nn.ModuleList = nn.ModuleList(BlockChunk(p) for p in chunked_blocks)
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm: nn.LayerNorm = norm_layer(embed_dim)
        self.head: nn.Module = nn.Identity()
        self.mask_token: nn.Parameter = nn.Parameter(torch.zeros(1, embed_dim))
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize model weights, positional embeddings, and register tokens."""
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(
        self,
        x: torch.Tensor,
        w: int,
        h: int,
    ) -> torch.Tensor:
        """Interpolate positional encodings for inputs whose resolution differs from training time."""
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        n_pos = self.pos_embed.shape[1] - 1
        if npatch == n_pos and w == h:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.patch_size
        h0 = h // self.patch_size
        m = int(math.sqrt(n_pos))
        if n_pos != m * m:
            err_msg = f"Expected {m * m} positional embeddings but got {n_pos}"
            raise ValueError(err_msg)

        kwargs: dict[str, object] = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / m
            sy = float(h0 + self.interpolate_offset) / m
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, m, m, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(
        self,
        x: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prepare input tokens with optional masking and positional encoding."""
        _, _, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1),
                self.mask_token.to(x.dtype).unsqueeze(0),
                x,
            )

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        return x

    def forward_features_list(
        self,
        x_list: list[torch.Tensor],
        masks_list: list[torch.Tensor | None],
    ) -> list[dict[str, torch.Tensor | None]]:
        """Compute forward features for a list of inputs with corresponding masks."""
        x: list[torch.Tensor] = [
            self.prepare_tokens_with_masks(x_item, masks) for x_item, masks in zip(x_list, masks_list, strict=True)
        ]
        for blk in self.blocks:
            x = blk(x)

        output: list[dict[str, torch.Tensor | None]] = []
        for x_item, masks in zip(x, masks_list, strict=True):
            x_norm = self.norm(x_item)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x_item,
                    "masks": masks,
                },
            )
        return output

    def forward_features(
        self,
        x: torch.Tensor | list[torch.Tensor],
        masks: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> dict[str, torch.Tensor | None] | list[dict[str, torch.Tensor | None]]:
        """Compute forward features for a single batch or list of batches."""
        if isinstance(x, list):
            masks_list: list[torch.Tensor | None]
            if masks is None:
                masks_list = [None] * len(x)
            elif isinstance(masks, list):
                masks_list = masks
            else:
                masks_list = [masks for _ in x]
            return self.forward_features_list(x, masks_list)

        features = self.prepare_tokens_with_masks(
            x,
            masks if isinstance(masks, torch.Tensor) else None,
        )

        for blk in self.blocks:
            features = blk(features)

        x_norm = self.norm(features)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": features,
            "masks": masks if isinstance(masks, torch.Tensor) else None,
        }

    def _get_intermediate_layers_not_chunked(
        self,
        x: torch.Tensor,
        n: int | Sequence[int] = 1,
    ) -> list[torch.Tensor]:
        """Extract intermediate outputs from specific layers when blocks are not chunked."""
        x = self.prepare_tokens_with_masks(x)
        output: list[torch.Tensor] = []
        total_block_len = len(self.blocks)
        blocks_to_take: range | Sequence[int]
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        return output

    def _get_intermediate_layers_chunked(
        self,
        x: torch.Tensor,
        n: int | Sequence[int] = 1,
    ) -> list[torch.Tensor]:
        """Extract intermediate outputs from specific layers when blocks are chunked."""
        x = self.prepare_tokens_with_masks(x)
        output: list[torch.Tensor] = []
        i = 0
        total_block_len = len(self.blocks[-1])
        blocks_to_take: range | Sequence[int]
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> tuple[torch.Tensor, ...] | tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Retrieve intermediate layer outputs.

        Args:
            x: Input tensor.
            n: Number of layers or explicit list of layer indices.
            reshape: Whether to reshape patch tokens into feature maps.
            return_class_token: Whether to include class tokens in the output.
            norm: Whether to apply final normalization.

        Returns:
            Tuple of intermediate outputs, optionally paired with class tokens.
        """
        outputs = (
            self._get_intermediate_layers_chunked(x, n)
            if self.chunked_blocks
            else self._get_intermediate_layers_not_chunked(x, n)
        )

        if norm:
            outputs = [self.norm(out) for out in outputs]

        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]

        if reshape:
            batch_size, _, w, h = x.shape
            outputs = [
                out.reshape(batch_size, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens, strict=True))
        return tuple(outputs)

    def forward(
        self,
        *args: object,
        is_training: bool = False,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | None] | torch.Tensor:
        """Apply the forward pass, returning classification output or full features."""
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        if isinstance(ret, list):
            msg = (
                "forward() received a list output in inference mode. "
                "List outputs are only supported in training mode (is_training=True). "
                "Inference mode requires a single tensor output. "
                "If you intended to get multiple outputs, please set is_training=True."
            )
            raise TypeError(msg)
        # inference: ret is a dict for non-list input
        return self.head(ret["x_norm_clstoken"])  # type: ignore[misc]


def init_weights_vit_timm(
    module: nn.Module,
    name: str = "",  # noqa: ARG001
) -> None:
    """Initialize module weights following the timm ViT initialization scheme."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(
    patch_size: int = 16,
    num_register_tokens: int = 0,
    **kwargs,
) -> DinoVisionTransformer:
    """Construct a small DINO Vision Transformer (ViT-S/16)."""
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_base(
    patch_size: int = 16,
    num_register_tokens: int = 0,
    **kwargs,
) -> DinoVisionTransformer:
    """Construct a base DINO Vision Transformer (ViT-B/16)."""
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_large(
    patch_size: int = 16,
    num_register_tokens: int = 0,
    **kwargs,
) -> DinoVisionTransformer:
    """Construct a large DINO Vision Transformer (ViT-L/16)."""
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_giant2(
    patch_size: int = 16,
    num_register_tokens: int = 0,
    **kwargs,
) -> DinoVisionTransformer:
    """Construct a Giant-2 DINO Vision Transformer variant."""
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
