# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model for the GeneralAD algorithm.

GeneralAD learns a patch-wise discriminator on top of frozen features extracted
from a pretrained backbone. During training it creates pseudo-anomalous patch
features using noise injection and patch shuffling/copying. During inference the
discriminator scores each patch and the highest patch scores are aggregated into
an image-level anomaly score.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import timm
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.data import InferenceBatch

if TYPE_CHECKING:
    from collections.abc import Sequence

FakeFeatureType = Literal[
    "random",
    "attn",
    "copy_out",
    "shuffle",
    "randshuffle",
    "copy_out_and_random",
    "copy_out_and_attn",
    "shuffle_and_random",
    "shuffle_and_attn",
    "randshuffle_and_random",
    "randshuffle_and_attn",
]

ATTENTION_REQUIRED_FAKE_FEATURE_TYPES = {
    "attn",
    "copy_out",
    "shuffle",
    "copy_out_and_attn",
    "shuffle_and_attn",
    "randshuffle_and_attn",
}


def _forward_attention_with_map(
    attn_obj: nn.Module,
    x: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Run timm attention while exposing the attention map."""
    del attn_mask, is_causal, kwargs
    batch_size, num_tokens, channels = x.shape
    qkv = attn_obj.qkv(x).reshape(
        batch_size,
        num_tokens,
        3,
        attn_obj.num_heads,
        channels // attn_obj.num_heads,
    )
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
    attn = attn.softmax(dim=-1)
    attn = attn_obj.attn_drop(attn)
    attn_obj.attn_map = attn

    x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
    return attn_obj.proj_drop(attn_obj.proj(x))


class ViTFeatureExtractor(nn.Module):
    """Extract patch features and class-attention maps from ViT backbones."""

    def __init__(
        self,
        backbone: str,
        layers: Sequence[int],
        image_size: tuple[int, int],
        pre_trained: bool = True,
    ) -> None:
        super().__init__()
        if image_size[0] != image_size[1]:
            msg = "GeneralAD currently expects square inputs for transformer backbones."
            raise ValueError(msg)

        if backbone.endswith("_ibot"):
            msg = (
                "iBOT checkpoints require external weights download and are not supported by this anomalib integration."
            )
            raise ValueError(msg)

        self.layers = sorted(set(layers))
        self.pretrained_model = timm.create_model(
            backbone,
            pretrained=pre_trained,
            num_classes=0,
            img_size=image_size[0],
        )
        for parameter in self.pretrained_model.parameters():
            parameter.requires_grad = False
        self.pretrained_model.eval()

        self.embed_dim = len(self.layers) * self.pretrained_model.embed_dim
        self.patch_size = int(self.pretrained_model.patch_embed.patch_size[0])
        self.num_patches = (image_size[0] // self.patch_size) ** 2
        self.start_index = getattr(self.pretrained_model, "num_prefix_tokens", 1)
        self.attn_layer = max(self.layers)

        if self.layers[0] < 1 or self.attn_layer > len(self.pretrained_model.blocks):
            msg = (
                f"Invalid layer indices {self.layers} for backbone '{backbone}'. "
                f"Valid transformer block indices are between 1 and {len(self.pretrained_model.blocks)}."
            )
            raise ValueError(msg)

        block = self.pretrained_model.blocks[self.attn_layer - 1]
        block.attn.forward = _forward_attention_with_map.__get__(block.attn, type(block.attn))

    def forward(
        self,
        images: torch.Tensor,
        output_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Extract transformer patch features and optional class attention."""
        x = self.pretrained_model.patch_embed(images)
        x = self.pretrained_model._pos_embed(x)  # noqa: SLF001
        x = self.pretrained_model.patch_drop(x)
        x = self.pretrained_model.norm_pre(x)

        outputs: list[torch.Tensor] = []
        for idx, layer in enumerate(self.pretrained_model.blocks, start=1):
            x = layer(x)
            if idx in self.layers:
                outputs.append(self.pretrained_model.norm(x[:, self.start_index :, :]))
            if idx == self.attn_layer:
                break

        features = torch.cat(outputs, dim=-1)
        if not output_attn:
            return features

        attn_map = self.pretrained_model.blocks[self.attn_layer - 1].attn.attn_map
        attn_map_cls = attn_map[:, :, 0, self.start_index :]
        return features, attn_map_cls


class EVAFeatureExtractor(nn.Module):
    """Extract patch features from EVA-style transformer backbones."""

    def __init__(
        self,
        backbone: str,
        layers: Sequence[int],
        image_size: tuple[int, int],
        pre_trained: bool = True,
    ) -> None:
        super().__init__()
        if image_size[0] != image_size[1]:
            msg = "GeneralAD currently expects square inputs for EVA backbones."
            raise ValueError(msg)

        self.layers = sorted(set(layers))
        self.pretrained_model = timm.create_model(
            backbone,
            pretrained=pre_trained,
            num_classes=0,
            img_size=image_size[0],
        )
        for parameter in self.pretrained_model.parameters():
            parameter.requires_grad = False
        self.pretrained_model.eval()

        self.embed_dim = len(self.layers) * self.pretrained_model.embed_dim
        self.patch_size = int(self.pretrained_model.patch_embed.patch_size[0])
        self.num_patches = (image_size[0] // self.patch_size) ** 2
        self.last_layer = max(self.layers)

        if self.layers[0] < 1 or self.last_layer > len(self.pretrained_model.blocks):
            msg = (
                f"Invalid layer indices {self.layers} for backbone '{backbone}'. "
                f"Valid transformer block indices are between 1 and {len(self.pretrained_model.blocks)}."
            )
            raise ValueError(msg)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract transformer patch features for EVA backbones."""
        x = self.pretrained_model.patch_embed(images)
        x, rot_pos_embed = self.pretrained_model._pos_embed(x)  # noqa: SLF001

        outputs: list[torch.Tensor] = []
        for idx, layer in enumerate(self.pretrained_model.blocks, start=1):
            x = layer(x, rope=rot_pos_embed)
            if idx in self.layers:
                outputs.append(self.pretrained_model.norm(x[:, 1:, :]))
            if idx == self.last_layer:
                break

        return torch.cat(outputs, dim=-1)


class AttentionBlock(nn.Module):
    """Transformer block used in the patch discriminator."""

    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and MLP updates to discriminator features."""
        attn_output, _ = self.attn(x, x, x)
        x = self.layer_norm(x + self.dropout1(attn_output))
        return x + self.dropout2(self.linear(x))


class PatchDiscriminator(nn.Module):
    """Attention-based patch discriminator."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_patches: int,
        num_layers: int = 1,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.transformer_encoder = nn.Sequential(
            *[AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout_rate) for _ in range(num_layers)],
        )
        self.output_layer = nn.Linear(embed_dim, 1, bias=False)
        self.positional_encodings = nn.Parameter(torch.randn(num_patches, embed_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Score each image patch with the discriminator."""
        features = features + self.positional_encodings.unsqueeze(0)
        features = self.transformer_encoder(features)
        return self.output_layer(features).squeeze(-1)


class GeneralADModel(nn.Module):
    """Core GeneralAD model."""

    def __init__(
        self,
        backbone: str = "vit_large_patch14_dinov2.lvd142m",
        layers: Sequence[int] = (24,),
        hidden_dim: int = 2048,
        noise_std: float = 0.25,
        dsc_layers: int = 1,
        dsc_heads: int = 4,
        dsc_dropout: float = 0.1,
        image_size: tuple[int, int] = (518, 518),
        num_fake_patches: int = -1,
        fake_feature_type: FakeFeatureType = "random",
        top_k: int = 10,
        pre_trained: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.layers = tuple(sorted(set(layers)))
        self.noise_std = noise_std
        self.fake_feature_type = fake_feature_type
        self.image_size = image_size

        if backbone.startswith("eva") and fake_feature_type in ATTENTION_REQUIRED_FAKE_FEATURE_TYPES:
            msg = (
                f"fake_feature_type='{fake_feature_type}' requires class-attention maps, "
                "but EVA backbones do not expose attention maps in this integration."
            )
            raise ValueError(msg)

        if backbone.startswith("vit"):
            self.feature_extractor = ViTFeatureExtractor(backbone, self.layers, image_size, pre_trained=pre_trained)
            self.attn_output = True
        elif backbone.startswith("eva"):
            self.feature_extractor = EVAFeatureExtractor(backbone, self.layers, image_size, pre_trained=pre_trained)
            self.attn_output = False
        else:
            msg = (
                f"Unsupported backbone '{backbone}'. GeneralAD currently supports "
                "transformer backbones starting with 'vit' or 'eva'."
            )
            raise ValueError(msg)

        self.patch_size = self.feature_extractor.patch_size
        self.num_patches = self.feature_extractor.num_patches
        self.patches_per_side = int(math.sqrt(self.num_patches))
        if top_k < -1 or top_k == 0:
            msg = "top_k must be -1 or a positive integer."
            raise ValueError(msg)
        if num_fake_patches < -1 or num_fake_patches == 0:
            msg = "num_fake_patches must be -1 or a positive integer."
            raise ValueError(msg)
        self.top_k = self.num_patches if top_k < 0 or top_k > self.num_patches else top_k
        self.num_fake_patches = (
            self.num_patches if num_fake_patches < 0 or num_fake_patches > self.num_patches else num_fake_patches
        )

        self.discriminator = PatchDiscriminator(
            embed_dim=self.feature_extractor.embed_dim,
            hidden_dim=hidden_dim,
            num_patches=self.num_patches,
            num_layers=dsc_layers,
            num_heads=dsc_heads,
            dropout_rate=dsc_dropout,
        )

    def extract_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Extract patch features and optional class-attention maps."""
        with torch.no_grad():
            if self.attn_output:
                features, attn_map = self.feature_extractor(images, output_attn=True)
                return features, attn_map
            return self.feature_extractor(images), None

    def score_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert patch logits to image and pixel anomaly scores."""
        patch_scores = self.discriminator(features)
        topk_values, _ = torch.topk(patch_scores, self.top_k, dim=1)
        image_scores = torch.mean(topk_values, dim=1)
        anomaly_map = patch_scores.reshape(-1, 1, self.patches_per_side, self.patches_per_side)
        anomaly_map = F.interpolate(anomaly_map, size=self.image_size, mode="bilinear", align_corners=False)
        return image_scores, anomaly_map

    def forward(self, images: torch.Tensor) -> InferenceBatch:
        """Predict anomaly scores and maps."""
        features, _ = self.extract_features(images)
        pred_score, anomaly_map = self.score_features(features)
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    def compute_loss(self, images: torch.Tensor) -> torch.Tensor:
        """Compute the GeneralAD self-supervised training loss."""
        features, attn_map = self.extract_features(images)
        loss = torch.tensor(0.0, device=images.device)

        scores_true = self.discriminator(features).flatten()
        masks_true = torch.zeros_like(scores_true)
        loss = loss + F.binary_cross_entropy_with_logits(scores_true, masks_true)

        fake_features, masks_fake = self._add_noise_all(features)
        scores_fake = self.discriminator(fake_features).flatten()
        loss = loss + F.binary_cross_entropy_with_logits(scores_fake, masks_fake.flatten())

        if self.fake_feature_type in {"random", "copy_out_and_random", "shuffle_and_random", "randshuffle_and_random"}:
            random_features, masks_random = self._add_random_noise(features)
            scores_random = self.discriminator(random_features).flatten()
            loss = loss + self._masked_bce(scores_random, masks_random)
        elif self.fake_feature_type in {"attn", "copy_out_and_attn", "shuffle_and_attn", "randshuffle_and_attn"}:
            if attn_map is None:
                msg = (
                    f"Fake feature type '{self.fake_feature_type}' requires a transformer backbone with attention maps."
                )
                raise ValueError(msg)
            attn_features, masks_attn = self._add_attn_noise(features, attn_map)
            scores_attn = self.discriminator(attn_features).flatten()
            loss = loss + self._masked_bce(scores_attn, masks_attn)

        if self.fake_feature_type in {"copy_out", "copy_out_and_random", "copy_out_and_attn"}:
            if attn_map is None:
                msg = (
                    f"Fake feature type '{self.fake_feature_type}' requires a transformer backbone with attention maps."
                )
                raise ValueError(msg)
            copy_features, masks_copy = self._add_attn_copy_out(features, attn_map)
            scores_copy = self.discriminator(copy_features).flatten()
            loss = loss + self._masked_bce(scores_copy, masks_copy)
        elif self.fake_feature_type in {"shuffle", "shuffle_and_random", "shuffle_and_attn"}:
            if attn_map is None:
                msg = (
                    f"Fake feature type '{self.fake_feature_type}' requires a transformer backbone with attention maps."
                )
                raise ValueError(msg)
            shuffle_features, masks_shuffle = self._add_attn_shuffle(features, attn_map)
            scores_shuffle = self.discriminator(shuffle_features).flatten()
            loss = loss + self._masked_bce(scores_shuffle, masks_shuffle)
        elif self.fake_feature_type in {"randshuffle", "randshuffle_and_random", "randshuffle_and_attn"}:
            randshuffle_features, masks_randshuffle = self._add_random_shuffle(features)
            scores_randshuffle = self.discriminator(randshuffle_features).flatten()
            loss = loss + self._masked_bce(scores_randshuffle, masks_randshuffle)

        return loss

    @staticmethod
    def _masked_bce(scores: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute BCE on normal and anomalous patches separately."""
        masks = masks.flatten()
        loss = torch.tensor(0.0, device=scores.device)
        if (~masks).any():
            loss = loss + F.binary_cross_entropy_with_logits(scores[~masks], masks[~masks].float())
        if masks.any():
            loss = loss + F.binary_cross_entropy_with_logits(scores[masks], masks[masks].float())
        return loss

    def _sample_patch_count(self, max_patches: int) -> int:
        return int(
            torch.randint(
                1,
                max_patches + 1,
                (1,),
                device=self.discriminator.positional_encodings.device,
            ).item(),
        )

    def _add_random_noise(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fake_features = features.clone()
        noise = torch.normal(0.0, self.noise_std, features.shape, device=features.device)
        batch_size, num_patches, _ = features.shape
        masks = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=features.device)

        for batch_idx in range(batch_size):
            num_fake = self._sample_patch_count(num_patches)
            indices = torch.randperm(num_patches, device=features.device)[:num_fake]
            masks[batch_idx, indices] = True
            fake_features[batch_idx, indices, :] += noise[batch_idx, indices, :]
        return fake_features, masks

    def _add_attn_noise(self, features: torch.Tensor, attn_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fake_features = features.clone()
        noise = torch.normal(0.0, self.noise_std, features.shape, device=features.device)
        batch_size, num_heads, num_patches = attn_map.shape
        masks = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=features.device)

        for batch_idx in range(batch_size):
            head = int(torch.randint(0, num_heads, (1,), device=features.device).item())
            num_fake = self._sample_patch_count(num_patches)
            indices = torch.topk(attn_map[batch_idx, head, :], num_fake).indices
            masks[batch_idx, indices] = True
            fake_features[batch_idx, indices, :] += noise[batch_idx, indices, :]
        return fake_features, masks

    def _add_attn_copy_out(self, features: torch.Tensor, attn_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fake_features = features.clone()
        batch_size, num_heads, num_patches = attn_map.shape
        masks = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=features.device)

        for batch_idx in range(batch_size):
            head = int(torch.randint(0, num_heads, (1,), device=features.device).item())
            num_fake = self._sample_patch_count(self.num_fake_patches)
            indices = torch.topk(attn_map[batch_idx, head, :], num_fake).indices
            random_indices = torch.randperm(num_patches, device=features.device)[:num_fake]
            masks[batch_idx, indices] = True
            fake_features[batch_idx, indices, :] = features[batch_idx, random_indices, :]
        return fake_features, masks

    def _add_attn_shuffle(self, features: torch.Tensor, attn_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fake_features = features.clone()
        batch_size, num_heads, _ = attn_map.shape
        masks = torch.zeros((batch_size, self.num_patches), dtype=torch.bool, device=features.device)

        for batch_idx in range(batch_size):
            head = int(torch.randint(0, num_heads, (1,), device=features.device).item())
            num_fake = self._sample_patch_count(self.num_fake_patches)
            indices = torch.topk(attn_map[batch_idx, head, :], num_fake).indices
            masks[batch_idx, indices] = True
            shuffled = fake_features[batch_idx, indices].clone()
            shuffled = shuffled[torch.randperm(num_fake, device=features.device)]
            fake_features[batch_idx, indices, :] = shuffled
        return fake_features, masks

    def _add_random_shuffle(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fake_features = features.clone()
        batch_size, num_patches, _ = features.shape
        masks = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=features.device)

        for batch_idx in range(batch_size):
            num_fake = self._sample_patch_count(self.num_fake_patches)
            indices = torch.randperm(num_patches, device=features.device)[:num_fake]
            masks[batch_idx, indices] = True
            shuffled = fake_features[batch_idx, indices].clone()
            shuffled = shuffled[torch.randperm(num_fake, device=features.device)]
            fake_features[batch_idx, indices, :] = shuffled
        return fake_features, masks

    def _add_noise_all(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.normal(0.0, self.noise_std, features.shape, device=features.device)
        fake_features = features + noise
        masks = torch.ones(features.shape[:2], dtype=features.dtype, device=features.device)
        return fake_features, masks
