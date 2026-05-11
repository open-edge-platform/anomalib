# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RADIO Model for AnomalyVFM."""

import math

import torch
from torch import nn
from torch.nn import functional


class Im2Patches(nn.Module):
    """Patch embedding module.

    Args:
        patch_size (int): patch size
    """

    def __init__(self, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify input image.

        Args:
            x: input image

        Returns:
            (torch.Tensor): patches
        """
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(b, c, h // p, w // p, p, p)
        return x.permute(0, 2, 3, 1, 4, 5).contiguous().view(b, -1, c * p * p)


class ViTPatchLinear(nn.Linear):
    """Patch embedding linear projection.

    Args:
        in_features (int): number of incoming features
        out_features (int): number of outcoming features
    """

    def __init__(self, in_features: int = 768, out_features: int = 1024, bias: bool = False) -> None:
        super().__init__(in_features, out_features, bias=bias)


class ClsToken(nn.Module):
    """[CLS] Token module.

    Args:
        ndim (int): dimension of [CLS] tokens
        num_tokens (int): number of [CLS] tokens
        enabled (bool): flag to use patch tokens
    """

    def __init__(self, ndim: int, num_tokens: int = 8, enabled: bool = True) -> None:
        super().__init__()
        self.ndim = ndim
        self.enabled = enabled
        self.num_tokens = num_tokens

        if enabled:
            scale = ndim**-0.5
            self.token = nn.Parameter(torch.randn(num_tokens, ndim) * scale)
        else:
            self.token = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add [CLS] tokens.

        Args:
            x: input features

        Returns:
            (torch.Tensor): [CLS] tokens + input features
        """
        if self.token is None:
            return x
        token = self.token.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([token, x], dim=1)


class ViTPatchGenerator(nn.Module):
    """Vision Transformer patch generator module.

    Args:
        patch_size (int): Size of the image patches. Default is 16.
        in_chans (int): Number of input image channels. Default is 3.
        embed_dim (int): Dimension of the embedding. Default is 1024.
        num_prefix_tokens (int): Number of prefix (e.g., [CLS]) tokens. Default is 8.
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        num_prefix_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.im_to_patches = Im2Patches(patch_size)
        self.embedder = ViTPatchLinear(
            in_features=in_chans * patch_size * patch_size,
            out_features=embed_dim,
            bias=False,
        )
        self.cls_token = ClsToken(ndim=embed_dim, num_tokens=num_prefix_tokens)
        self.patch_normalizer = nn.Identity()

        self.pos_embed = nn.Parameter(torch.zeros(1, 16384, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate embedded patches with prefix tokens.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            (torch.Tensor): Token embeddings with [CLS] tokens attached.
        """
        x = self.im_to_patches(x)
        x = self.embedder(x)
        x = self.cls_token(x)
        return self.patch_normalizer(x)


class Mlp(nn.Module):
    """Multilayer Perceptron module.

    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features.
        out_features (int): Number of output features.
        act_layer (Type[nn.Module]): Activation layer type. Default is nn.GELU.
        drop (float): Dropout probability. Default is 0.0.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer(approximate="none")
        self.drop1 = nn.Dropout(drop)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP to input features.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            (torch.Tensor): MLP output features.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)


class Attention(nn.Module):
    """Multi-head self-attention module.

    Args:
        dim (int): Input feature dimension.
        num_heads (int): Number of attention heads. Default is 16.
        qkv_bias (bool): Flag to include bias in QKV projections. Default is True.
        attn_drop (float): Dropout probability for attention weights. Default is 0.0.
        proj_drop (float): Dropout probability for output projection. Default is 0.0.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention.

        Args:
            x (torch.Tensor): Input sequence features.

        Returns:
            (torch.Tensor): Attended features.
        """
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        x = functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.norm(x)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    """Transformer block module.

    Args:
        dim (int): Input feature dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dimension to input dimension. Default is 4.0.
        qkv_bias (bool): Flag to include bias in QKV projections. Default is True.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-06)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.ls1 = nn.Identity()
        self.drop_path1 = nn.Identity()

        self.norm2 = nn.LayerNorm(dim, eps=1e-06)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim)
        self.ls2 = nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a single transformer block.

        Args:
            x (torch.Tensor): Input sequence features.

        Returns:
            (torch.Tensor): Block output features.
        """
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))


class VisionTransformer(nn.Module):
    """Vision Transformer backbone module.

    Args:
        patch_size (int): Size of the image patches. Default is 16.
        embed_dim (int): Dimension of the token embeddings. Default is 1024.
        depth (int): Number of transformer blocks. Default is 24.
        num_heads (int): Number of attention heads. Default is 16.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension. Default is 4.0.
        num_prefix_tokens (int): Number of prefix tokens. Default is 8.
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_prefix_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_prefix_tokens = num_prefix_tokens

        self.patch_embed = None
        self.pos_drop = None
        self.patch_drop = nn.Identity()
        self.norm_pre = nn.Identity()

        self.patch_generator = ViTPatchGenerator(
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_prefix_tokens=num_prefix_tokens,
        )

        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.norm = nn.Identity()
        self.fc_norm = nn.Identity()
        self.head_drop = nn.Dropout(0.0)
        self.head = nn.Identity()

    def _interpolate_pos_encoding(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Interpolate positional encodings for dynamic image resolutions.

        Args:
            x (torch.Tensor): Input features.
            h (int): Original image height.
            w (int): Original image width.

        Returns:
            (torch.Tensor): Interpolated positional encodings.
        """
        pos_embed = self.patch_generator.pos_embed
        seq_length = pos_embed.shape[1]

        if int(math.sqrt(seq_length - self.num_prefix_tokens)) ** 2 == seq_length - self.num_prefix_tokens:
            pos_prefix_len = self.num_prefix_tokens
        else:
            base_grid = int(math.sqrt(seq_length))
            pos_prefix_len = seq_length - (base_grid**2)

        base_spatial_len = seq_length - pos_prefix_len
        base_grid = int(math.sqrt(base_spatial_len))

        extra_pos_embed = pos_embed[:, :pos_prefix_len]
        patch_pos_embed = pos_embed[:, pos_prefix_len:]

        dim = x.shape[-1]
        w0, h0 = w // self.patch_size, h // self.patch_size

        if h0 != base_grid or w0 != base_grid:
            patch_pos_embed = patch_pos_embed.reshape(1, base_grid, base_grid, dim).permute(0, 3, 1, 2)
            patch_pos_embed = functional.interpolate(
                patch_pos_embed,
                size=(h0, w0),
                mode="bicubic",
                align_corners=False,
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        if pos_prefix_len < self.num_prefix_tokens:
            padding = torch.zeros(
                1,
                self.num_prefix_tokens - pos_prefix_len,
                dim,
                device=pos_embed.device,
                dtype=pos_embed.dtype,
            )
            extra_pos_embed = torch.cat((extra_pos_embed, padding), dim=1)

        elif pos_prefix_len > self.num_prefix_tokens:
            extra_pos_embed = extra_pos_embed[:, : self.num_prefix_tokens]

        return torch.cat((extra_pos_embed, patch_pos_embed), dim=1)

    def forward(self, x: torch.Tensor, original_h: int, original_w: int) -> torch.Tensor:
        """Forward pass through the Vision Transformer.

        Args:
            x (torch.Tensor): Input image tensor.
            original_h (int): Original height of the input image.
            original_w (int): Original width of the input image.

        Returns:
            (torch.Tensor): Output features from the final transformer block.
        """
        x = self.patch_generator(x)
        x = self.norm_pre(x)
        x = x + self._interpolate_pos_encoding(x, original_h, original_w)
        x = self.patch_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return self.head(x)


class InputConditioner(nn.Module):
    """Input conditioning module for image normalization."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("norm_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1))
        self.register_buffer("norm_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input image tensor using standard ImageNet mean and std.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            (torch.Tensor): Normalized image tensor.
        """
        return (x - self.norm_mean) / self.norm_std


class RADIOModel(nn.Module):
    """RADIO framework model module.

    Args:
        num_prefix_tokens (int): Number of prefix tokens. Default is 8.
    """

    def __init__(self, num_prefix_tokens: int = 8) -> None:
        super().__init__()
        self.model = VisionTransformer(num_prefix_tokens=num_prefix_tokens)
        self.input_conditioner = InputConditioner()
        self.adaptors = nn.ModuleDict()
        self.feature_normalizer = nn.Identity()

        self.register_buffer("summary_idxs", torch.tensor([0, 1, 2]))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass extracting summary and spatial features.

        Args:
            x (torch.Tensor): Input image tensor.
            feature_fmt (str): Expected format of the spatial features. Default is "NLD".

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): A tuple containing:
                - summary (torch.Tensor): Flattened summary features from designated tokens.
                - spatial_features (torch.Tensor): Spatial patch features.
        """
        b, c, h, w = x.shape
        x = self.input_conditioner(x)
        x = self.model(x, original_h=h, original_w=w)
        x = self.feature_normalizer(x)

        summary = x[:, self.summary_idxs]
        spatial_features = x[:, self.model.num_prefix_tokens :]

        b, c, d = summary.shape
        summary = summary.reshape(b, c * d)
        return summary, spatial_features
