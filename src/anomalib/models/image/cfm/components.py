# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Net components for CFM (Extractors, Mappers, Transformers)."""

from __future__ import annotations

import timm
import torch
from timm.models.layers import DropPath
from torch import nn
from torch.nn import functional

from .utils import farthest_point_sample, index_points, interpolating_points


class FeatureProjectionMLP(nn.Module):
    """MLP that learns to map feature between domains(2D <-> 3D)."""

    def __init__(self, in_features: int, out_features: int, act_layer: type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        hidden_dim = (in_features + out_features) // 2
        self.act_fcn = act_layer()
        self.input = nn.Conv2d(in_features, hidden_dim, kernel_size=1)
        self.projection = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.output = nn.Conv2d(hidden_dim, out_features, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.act_fcn(self.input(x))
        x = self.act_fcn(self.projection(x))
        return self.output(x)


class MultimodalFeatures(nn.Module):
    """It manages the DINO and Point-MAE backbones and synchronises their features."""

    def __init__(
        self,
        image_size: int = 224,
        rgb_backbone_name: str = "vit_base_patch8_224.dino",
        group_size: int = 128,
        num_group: int = 1024,
    ) -> None:
        super().__init__()

        # Uploades wrapper with DINO and Point-MAE
        self.extractors = FeatureExtractors(
            rgb_backbone_name=rgb_backbone_name,
            group_size=group_size,
            num_group=num_group,
        )

        self.image_size = image_size

        # Resize and smoothing
        self.resize = nn.AdaptiveAvgPool2d((image_size, image_size))
        self.average = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def get_features_maps(self, rgb: torch.Tensor, pc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract 2D and 3D features, interpolates and aligns them dimensionally."""
        # pc -> (B, C, H, W). Transformed in (B, N, 3)
        b = pc.shape[0]
        unorganized_pc = pc.view(b, pc.shape[1], -1).permute(0, 2, 1)

        # Removed points (0,0,0) to avoid noise.
        nonzero_indices = torch.nonzero(torch.all(unorganized_pc[0] != 0, dim=1)).squeeze(dim=1)
        unorganized_pc_no_zeros = unorganized_pc[:, nonzero_indices, :]

        # Extraction (only eval/no_grad)
        with torch.no_grad():
            rgb_raw, xyz_raw, center, _, _ = self.extractors(rgb, unorganized_pc_no_zeros)

        # Interpolate the reduced centres of the PointTransformer onto the original points
        interpolated_pc = interpolating_points(
            unorganized_pc_no_zeros.transpose(1, 2),
            center.permute(0, 2, 1),
            xyz_raw,
        )

        # Feature 3D onto 2D grid
        xyz_patch_full = torch.zeros(
            (b, interpolated_pc.shape[1], self.image_size * self.image_size),
            dtype=interpolated_pc.dtype,
            device=rgb.device,
        )
        xyz_patch_full[..., nonzero_indices] = interpolated_pc

        # Smooth and Resize for XYZ
        xyz_patch_full_2d = xyz_patch_full.view(b, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_feat = self.resize(self.average(xyz_patch_full_2d))

        # Resize for RGB
        rgb_feat = functional.interpolate(
            rgb_raw,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        rgb_feat = self.average(rgb_feat)

        return rgb_feat, xyz_feat


class FeatureExtractors(nn.Module):
    """Wrapper for the two pre-trained Backbone."""

    def __init__(self, rgb_backbone_name: str, group_size: int = 128, num_group: int = 1024) -> None:
        super().__init__()
        layers_keep = 12

        # RGB Backbone (DINO via timm)
        self.rgb_backbone = timm.create_model(model_name=rgb_backbone_name, pretrained=True, features_only=False)
        self.rgb_backbone.blocks = nn.Sequential(*self.rgb_backbone.blocks[:layers_keep])

        # XYZ Backbone (PointTransformer)
        self.xyz_backbone = PointTransformer(group_size=group_size, num_group=num_group)

        try:
            self.xyz_backbone.load_model_from_ckpt("checkpoints/feature_extractors/pointmae_pretrain.pth")
        except FileNotFoundError:
            print("Attention: Point-MAE weights not found. Random weights will be used.")
        self.xyz_backbone.blocks.blocks = nn.Sequential(*self.xyz_backbone.blocks.blocks[:layers_keep])

    def forward_rgb_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts RGB features."""
        x = self.rgb_backbone.forward_features(x)
        # Patch size 8x8 on 224x224 = grid 28x28
        return x[:, 1:].permute(0, 2, 1).view(x.shape[0], -1, 28, 28)

    def forward(
        self,
        rgb: torch.Tensor,
        xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        rgb_features = self.forward_rgb_features(rgb)
        xyz_features, center, ori_idx, center_idx = self.xyz_backbone(xyz.transpose(1, 2))
        return rgb_features, xyz_features, center, ori_idx, center_idx


def fps(data: torch.Tensor, number: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sampling FPS."""
    fps_idx = farthest_point_sample(data, number)
    fps_data = index_points(data, fps_idx)
    return fps_data, fps_idx


class KNN(nn.Module):
    """K-Nearest Neighbors module."""

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def forward(self, xyz: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        xyz = xyz.unsqueeze(2)
        centers = centers.unsqueeze(1)
        distances = torch.norm(xyz - centers, dim=-1)
        _, indices = torch.topk(distances, self.k, dim=1, largest=False, sorted=True)
        return indices


class Group(nn.Module):
    """Groups points into local neighborhoods."""

    def __init__(self, num_group: int, group_size: int) -> None:
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size)

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        batch_size, num_points, _ = xyz.shape
        center, center_idx = fps(xyz.contiguous(), self.num_group)
        idx = self.knn(xyz, center).permute(0, 2, 1)

        ori_idx = idx
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.reshape(-1)

        neighborhood = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, ori_idx, center_idx


class Encoder(nn.Module):
    """Encoder module."""

    def __init__(self, encoder_channel: int) -> None:
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class MLP(nn.Module):
    """Multilayer Perceptron module."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    """Attention module."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.attn_drop(attn.softmax(dim=-1))
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(b, n, c)))


class Block(nn.Module):
    """Transformer Block module."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.attn = Attention(dim, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class TransformerEncoder(nn.Module):
    """Transformer Encoder module."""

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 4,
        num_heads: int = 12,
        drop_path_rate: float | list[float] = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass."""
        feature_list = []
        for block in self.blocks:
            x = block(x + pos)
            feature_list.append(x)
        return feature_list


class PointTransformer(nn.Module):
    """Point Transformer module."""

    def __init__(self, group_size: int = 128, num_group: int = 1024, encoder_dims: int = 384) -> None:
        super().__init__()
        self.trans_dim = 384
        self.depth = 12
        self.group_size = group_size
        self.num_group = num_group

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder_dims = encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim))
        dpr = [x.item() for x in torch.linspace(0, 0.1, self.depth)]
        self.blocks = TransformerEncoder(embed_dim=self.trans_dim, depth=self.depth, drop_path_rate=dpr, num_heads=6)
        self.norm = nn.LayerNorm(self.trans_dim)

    def load_model_from_ckpt(self, bert_ckpt_path: str) -> None:
        """Loads model from a checkpoint."""
        ckpt = torch.load(bert_ckpt_path, map_location="cpu" if not torch.cuda.is_available() else "cuda")  # nosec B614
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt["base_model"].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith("MAE_encoder."):
                base_ckpt[k[len("MAE_encoder.") :]] = base_ckpt[k]
                del base_ckpt[k]
            elif k.startswith("base_model."):
                base_ckpt[k[len("base_model.") :]] = base_ckpt[k]
                del base_ckpt[k]
        self.load_state_dict(base_ckpt, strict=False)

    def forward(self, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        pts = pts.transpose(-1, -2)
        neighborhood, center, ori_idx, center_idx = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)
        pos = self.pos_embed(center)
        feature_list = self.blocks(group_input_tokens, pos)
        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        x = torch.cat((feature_list[3], feature_list[7], feature_list[11]), dim=1)
        return x, center, ori_idx, center_idx
