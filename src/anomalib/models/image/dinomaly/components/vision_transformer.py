# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Vision Transformer implementation for Dinomaly model.

This module contains the implementation of the Vision Transformer architecture used in the Dinomaly model.
Use the methods `vit_small`, `vit_base`, `vit_large`, and `vit_giant2` to create a DinoVisionTransformer instance
that can be used as an encoder for the Dinomaly model.


References:
    https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
    https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py
"""

import logging
import math
from collections.abc import Callable
from functools import partial

import torch
import torch.utils.checkpoint
from timm.layers.patch_embed import PatchEmbed
from torch import nn
from torch.nn.init import trunc_normal_

from anomalib.models.image.dinomaly.components.layers import Block, DinomalyMLP, MemEffAttention

logger = logging.getLogger("dinov2")


def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    """Apply a function recursively to module and its children.

    Args:
        fn: Function to apply to each module
        module: Root module to apply function to
        name: Name of the current module
        depth_first: Whether to apply function depth-first
        include_root: Whether to include the root module

    Returns:
        The modified module
    """
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        named_apply(fn=fn, module=child_module, name=full_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    """A chunk of transformer blocks for efficient processing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all blocks in the chunk."""
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    """DINOv2 Vision Transformer implementation for anomaly detection."""

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
        init_values: float | None = None,  # for layerscale: None or 0 => no layerscale
        embed_layer: type = PatchEmbed,
        act_layer: type = nn.GELU,
        block_fn: type | partial[Block] = Block,
        block_chunks: int = 1,
        num_register_tokens: int = 0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
    ) -> None:
        """Initialize the DINOv2 Vision Transformer.

        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            strict_img_size=False,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks_list = [
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
                ffn_layer=DinomalyMLP,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunksize = depth // block_chunks
            chunked_blocks = [[nn.Identity()] * i + blocks_list[i : i + chunksize] for i in range(0, depth, chunksize)]
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights of the vision transformer."""
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        """Interpolate positional encodings for different input sizes.

        Args:
            x: Input tensor
            w: Width of the input
            h: Height of the input

        Returns:
            Interpolated positional encodings
        """
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
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0_float = float(w0) + self.interpolate_offset
        h0_float = float(h0) + self.interpolate_offset

        sqrt_n = math.sqrt(n_pos)
        sx, sy = w0_float / sqrt_n, h0_float / sqrt_n
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_n), int(sqrt_n), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )

        assert int(w0_float) == patch_pos_embed.shape[-2]
        assert int(h0_float) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        """Prepare tokens for transformer input with optional masking.

        Args:
            x: Input tensor
            masks: Optional mask tensor

        Returns:
            Prepared tokens with positional encoding
        """
        _batch_size, _num_channels, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

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

    def prepare_tokens(self, x: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        """Prepare tokens for transformer input.

        Args:
            x: Input tensor
            masks: Optional mask tensor

        Returns:
            Prepared tokens with positional encoding
        """
        _batch_size, _num_channels, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the vision transformer.

        Args:
            x: Input tensor

        Returns:
            CLS token output
        """
        x = self.prepare_tokens_with_masks(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]

    def get_intermediate_layers(self, x: torch.Tensor, n: int = 1) -> list[torch.Tensor]:
        """Get outputs from the last n transformer blocks.

        Args:
            x: Input tensor
            n: Number of last blocks to get outputs from

        Returns:
            List of outputs from the last n blocks
        """
        x = self.prepare_tokens_with_masks(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

    def get_last_selfattention(self, x: torch.Tensor) -> torch.Tensor:
        """Get self-attention from the last transformer block.

        Args:
            x: Input tensor

        Returns:
            Self-attention tensor from the last block
        """
        x = self.prepare_tokens_with_masks(x)
        for i, block in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = block(x)
            else:
                # return attention of the last block
                return block(x, return_attention=True)[:, self.num_register_tokens + 1 :]

        # This should never be reached, but added for type safety
        msg = "No blocks found in the transformer"
        raise RuntimeError(msg)

    def get_all_selfattention(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Get self-attention matrices from every transformer layer.

        Args:
            x: Input tensor

        Returns:
            List of self-attention tensors from all layers
        """
        x = self.prepare_tokens_with_masks(x)
        attns = []

        for block in self.blocks:
            attn = block(x, return_attention=True)
            attn = torch.cat([attn[:, :, :1, :], attn[:, :, self.num_register_tokens + 1 :, :]], dim=2)
            attn = torch.cat([attn[:, :, :, :1], attn[:, :, :, self.num_register_tokens + 1 :]], dim=3)

            attns.append(attn)
            x = block(x)

        return attns


def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:  # noqa: ARG001
    """ViT weight initialization, original timm impl (for reproducibility)."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size: int = 16, num_register_tokens: int = 0, **kwargs) -> DinoVisionTransformer:
    """Create a small Vision Transformer model.

    Args:
        patch_size: Size of image patches
        num_register_tokens: Number of register tokens
        **kwargs: Additional arguments

    Returns:
        DinoVisionTransformer model instance
    """
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


def vit_base(patch_size: int = 16, num_register_tokens: int = 0, **kwargs) -> DinoVisionTransformer:
    """Create a base Vision Transformer model.

    Args:
        patch_size: Size of image patches
        num_register_tokens: Number of register tokens
        **kwargs: Additional arguments

    Returns:
        DinoVisionTransformer model instance
    """
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


def vit_large(patch_size: int = 16, num_register_tokens: int = 0, **kwargs) -> DinoVisionTransformer:
    """Create a large Vision Transformer model.

    Args:
        patch_size: Size of image patches
        num_register_tokens: Number of register tokens
        **kwargs: Additional arguments

    Returns:
        DinoVisionTransformer model instance
    """
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


def vit_giant2(patch_size: int = 16, num_register_tokens: int = 0, **kwargs) -> DinoVisionTransformer:
    """Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64.

    Args:
        patch_size: Size of image patches
        num_register_tokens: Number of register tokens
        **kwargs: Additional arguments

    Returns:
        DinoVisionTransformer model instance
    """
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
