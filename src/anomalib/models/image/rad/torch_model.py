# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model for RAD (Retrieval-based Anomaly Detection).

This module implements the core RAD algorithm: multi-layer feature extraction,
memory bank construction, image-level retrieval, and position-aware patch-level
anomaly scoring.

Paper: https://arxiv.org/abs/2601.22763
"""

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.data import InferenceBatch
from anomalib.models.components import (
    DynamicBufferMixin,
    GaussianBlur2d,
    TimmFeatureExtractor,
)


class RadModel(DynamicBufferMixin, nn.Module):
    """RAD PyTorch model for anomaly detection.

    Implements training-free retrieval-based anomaly detection using multi-layer
    ViT features and position-aware patch matching.

    Args:
        backbone (str): Name of the ViT backbone.
            Defaults to ``"vit_base_patch16_dinov3"``.
        pre_trained (bool): Whether to use pre-trained backbone weights.
            Defaults to ``True``.
        layers (list[int]): Block indices (0-based) to extract features from.
            Defaults to ``[3, 6, 9, 11]``.
        k_image (int): Number of nearest-neighbor training images for local
            patch memory. Defaults to ``150``.
        use_positional_bank (bool): Enable position-aware patch matching.
            Defaults to ``True``.
        pos_radius (int): Spatial neighborhood radius (in patch units) for
            position-aware matching. Defaults to ``1``.
        max_ratio (float): Fraction of highest anomaly pixels pooled for
            image-level score. ``0`` means use max. Defaults to ``0.01``.
        layer_weights (list[float] | None): Weights for each layer in score
            fusion. ``None`` means uniform. Defaults to ``None``.

    Example:
        >>> model = RadModel()
        >>> x = torch.randn(2, 3, 448, 448)
        >>> model.training = True
        >>> _ = model(x)
    """

    def __init__(
        self,
        backbone: str = "vit_base_patch16_dinov3",
        pre_trained: bool = True,
        layers: list[int] | None = None,
        k_image: int = 150,
        use_positional_bank: bool = True,
        pos_radius: int = 1,
        max_ratio: float = 0.01,
        layer_weights: list[float] | None = None,
    ) -> None:
        super().__init__()

        if layers is None:
            layers = [3, 6, 9, 11]

        self.layers = layers
        self.k_image = k_image
        self.use_positional_bank = use_positional_bank
        self.pos_radius = pos_radius
        self.max_ratio = max_ratio

        layer_names = [f"blocks.{i}" for i in self.layers]
        self.feature_extractor = TimmFeatureExtractor(
            backbone=backbone,
            pre_trained=pre_trained,
            layers=layer_names,
            requires_grad=False,
            output_fmt="NLC",
            return_class_token=True,
            norm=True,
            dynamic_img_size=True,
        )

        num_layers = len(self.layers)
        if layer_weights is None:
            self._layer_weights = [1.0 / num_layers] * num_layers
        else:
            s = sum(layer_weights)
            self._layer_weights = [w / s for w in layer_weights]

        # Memory bank buffers populated during fit
        # cls_banks[i]: (N, C) - CLS tokens per layer
        # patch_banks[i]: (N, L, C) - patch tokens per layer
        self.cls_banks: list[torch.Tensor] = []
        self.patch_banks: list[torch.Tensor] = []

        # Stores for accumulating features during training
        self._cls_store: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
        self._patch_store: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]

        # Gaussian smoothing applied to anomaly map
        self.blur = GaussianBlur2d(kernel_size=(5, 5), sigma=(1.0, 1.0), channels=1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Process input through the model.

        During training, extracts and stores features. During inference, computes
        anomaly maps and scores via retrieval-based matching.

        Args:
            input_tensor (torch.Tensor): Input images of shape
                ``(B, 3, H, W)``.

        Returns:
            torch.Tensor | InferenceBatch: Embeddings during training, or
                InferenceBatch with anomaly maps and scores during inference.
        """
        features = self._extract_features(input_tensor)

        if self.training:
            for li, (patch_tok, cls_tok) in enumerate(features):
                self._cls_store[li].append(cls_tok.cpu())
                self._patch_store[li].append(patch_tok.cpu())
            return features[0][1]  # Return CLS for Lightning compatibility

        return self._score(features, input_tensor.shape[-2:])

    def _extract_features(
        self,
        x: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Extract multi-layer features from the backbone.

        Returns:
            List of (patch_tokens, cls_token) tuples per layer.
            patch_tokens: (B, L, C), cls_token: (B, C)
        """
        layer_names = [f"blocks.{i}" for i in self.layers]
        raw_features = self.feature_extractor(x)

        result = []
        num_prefix = self.feature_extractor.num_prefix_tokens
        for name in layer_names:
            feat = raw_features[name]  # (B, N_total, C) with prefix tokens
            cls_tok = feat[:, 0, :]  # (B, C) - CLS token
            patch_tok = feat[:, num_prefix:, :]  # (B, L, C) - patch tokens
            result.append((patch_tok, cls_tok))

        return result

    def build_memory_bank(self) -> None:
        """Build memory bank from accumulated training features."""
        num_layers = len(self.layers)
        self.cls_banks = []
        self.patch_banks = []

        for li in range(num_layers):
            cls_bank = torch.cat(self._cls_store[li], dim=0)  # (N, C)
            patch_bank = torch.cat(self._patch_store[li], dim=0)  # (N, L, C)
            self.cls_banks.append(cls_bank)
            self.patch_banks.append(patch_bank)

        # Clear stores
        self._cls_store = [[] for _ in range(num_layers)]
        self._patch_store = [[] for _ in range(num_layers)]

    @torch.no_grad()
    def _score(
        self,
        features: list[tuple[torch.Tensor, torch.Tensor]],
        output_size: tuple[int, int],
    ) -> InferenceBatch:
        """Compute anomaly maps and scores via retrieval-based matching.

        Args:
            features: Multi-layer features from test images.
            output_size: Original spatial size (H, W) for upsampling.

        Returns:
            InferenceBatch with anomaly_map and pred_score.
        """
        device = features[0][0].device
        num_layers = len(self.layers)

        # Move banks to device
        cls_banks_dev = [cb.to(device) for cb in self.cls_banks]
        patch_banks_dev = [pb.to(device) for pb in self.patch_banks]

        # Unpack features
        patch_list = [f[0] for f in features]  # list of (B, L, C)
        cls_list = [f[1] for f in features]  # list of (B, C)

        batch_size = patch_list[0].shape[0]
        num_patches = patch_list[0].shape[1]
        h = w = int(num_patches**0.5)

        # Image-level retrieval using highest layer CLS
        cls_query = F.normalize(cls_list[-1], dim=-1)  # (B, C)
        cls_bank = F.normalize(cls_banks_dev[-1], dim=-1)  # (N, C)
        sim_img = torch.matmul(cls_query, cls_bank.t())  # (B, N)
        k = min(self.k_image, sim_img.shape[1])
        _, topk_idx = torch.topk(sim_img, k, dim=-1)  # (B, k)

        # Multi-layer patch scoring
        patch_scores_batch = self._compute_patch_scores(
            patch_list=patch_list,
            patch_banks_dev=patch_banks_dev,
            topk_idx=topk_idx,
            h=h,
            w=w,
            num_layers=num_layers,
            device=device,
        )

        # Reshape and upsample to pixel-level
        patch_maps = patch_scores_batch.view(batch_size, 1, h, w)
        anomaly_map = F.interpolate(
            patch_maps,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        # Gaussian smoothing
        anomaly_map = self.blur(anomaly_map)

        # Image-level score
        if self.max_ratio == 0:
            pred_score = anomaly_map.flatten(1).max(dim=1)[0]
        else:
            amap_flat = anomaly_map.flatten(1)
            top_k = max(1, int(amap_flat.shape[1] * self.max_ratio))
            pred_score = torch.sort(amap_flat, dim=1, descending=True)[0][
                :,
                :top_k,
            ].mean(dim=1)

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    def _compute_patch_scores(
        self,
        patch_list: list[torch.Tensor],
        patch_banks_dev: list[torch.Tensor],
        topk_idx: torch.Tensor,
        h: int,
        w: int,
        num_layers: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute fused multi-layer patch anomaly scores.

        Args:
            patch_list: Test patch features per layer, each (B, L, C).
            patch_banks_dev: Memory bank patch features per layer, each (N, L, C).
            topk_idx: Indices of k nearest training images, (B, k).
            h: Patch grid height.
            w: Patch grid width.
            num_layers: Number of layers.
            device: Computation device.

        Returns:
            Fused patch scores of shape (B, L).
        """
        batch_size = patch_list[0].shape[0]
        num_patches = h * w

        scores_all = torch.zeros(batch_size, num_patches, device=device)

        for li in range(num_layers):
            weight = self._layer_weights[li]
            patches_x = F.normalize(patch_list[li], dim=-1)  # (B, L, C)
            bank = patch_banks_dev[li]  # (N, L_bank, C)

            layer_scores = self._score_layer(
                patches_x=patches_x,
                bank=bank,
                topk_idx=topk_idx,
                h=h,
                w=w,
                device=device,
            )
            scores_all += weight * layer_scores

        return scores_all

    def _score_layer(
        self,
        patches_x: torch.Tensor,
        bank: torch.Tensor,
        topk_idx: torch.Tensor,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Score patches for a single layer.

        Args:
            patches_x: Normalized test patches (B, L, C).
            bank: Patch bank for this layer (N, L_bank, C).
            topk_idx: Retrieved image indices (B, k).
            h: Patch grid height.
            w: Patch grid width.
            device: Computation device.

        Returns:
            Anomaly scores per patch (B, L).
        """
        if self.use_positional_bank:
            return self._score_layer_positional(patches_x, bank, topk_idx, h, w, device)

        # Batched global patch-KNN
        neigh_feat = F.normalize(bank[topk_idx], dim=-1)  # (B, k, L_bank, C)
        batch_size, k, l_bank, c = neigh_feat.shape
        bank_local = neigh_feat.reshape(batch_size, k * l_bank, c)  # (B, k*L_bank, C)
        sim = torch.bmm(patches_x, bank_local.transpose(1, 2))  # (B, L, k*L_bank)
        nn_sim = sim.max(dim=-1)[0]  # (B, L)
        return 1.0 - nn_sim

    def _score_layer_positional(
        self,
        patches_x: torch.Tensor,
        bank: torch.Tensor,
        topk_idx: torch.Tensor,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Score patches with position-aware matching.

        Each query patch only compares against patches from nearby spatial
        positions in the memory bank.
        """
        batch_size, num_patches, c = patches_x.shape

        # Padded neighborhood indices: (L, K_max) and mask (L, K_max)
        pos_idx, pos_mask = self._get_positional_indices(h, w, self.pos_radius, device)
        k_max = pos_idx.shape[1]

        neigh_feat = F.normalize(bank[topk_idx], dim=-1)  # (B, k, L_bank, C)
        k = neigh_feat.shape[1]

        # Gather neighbor patches for all positions at once
        # pos_idx: (L, K_max) → expand for (B, k, L*K_max, C)
        idx_flat = pos_idx.reshape(-1)  # (L*K_max,)
        neigh_gathered = neigh_feat[:, :, idx_flat, :]  # (B, k, L*K_max, C)
        neigh_gathered = neigh_gathered.reshape(batch_size, k, num_patches, k_max, c)
        # → (B, L, k*K_max, C)
        neigh_gathered = neigh_gathered.permute(0, 2, 1, 3, 4).reshape(batch_size, num_patches, k * k_max, c)

        # Batched similarity: (B, L, 1, C) @ (B, L, C, k*K_max) → (B, L, 1, k*K_max)
        sim = torch.matmul(patches_x.unsqueeze(2), neigh_gathered.transpose(2, 3)).squeeze(2)  # (B, L, k*K_max)

        # Mask out padded positions (repeat mask across k neighbors)
        mask_expanded = pos_mask.unsqueeze(0).expand(batch_size, -1, -1)  # (B, L, K_max)
        mask_expanded = mask_expanded.unsqueeze(2).expand(-1, -1, k, -1)  # (B, L, k, K_max)
        mask_expanded = mask_expanded.reshape(batch_size, num_patches, k * k_max)  # (B, L, k*K_max)

        sim = sim.masked_fill(~mask_expanded, -1.0)
        nn_sim = sim.max(dim=-1)[0]  # (B, L)
        return 1.0 - nn_sim

    @staticmethod
    def _get_positional_indices(
        h: int,
        w: int,
        radius: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute padded spatial neighborhood indices for each patch.

        Args:
            h: Grid height.
            w: Grid width.
            radius: Neighborhood radius.
            device: Device for tensors.

        Returns:
            Tuple of (indices, mask) both of shape (L, K_max).
        """
        k_max = (2 * radius + 1) ** 2
        num_patches = h * w
        pos_idx = torch.zeros(num_patches, k_max, dtype=torch.long, device=device)
        pos_mask = torch.zeros(num_patches, k_max, dtype=torch.bool, device=device)

        for j in range(num_patches):
            r = j // w
            col = j % w

            r_min = max(0, r - radius)
            r_max = min(h - 1, r + radius)
            c_min = max(0, col - radius)
            c_max = min(w - 1, col + radius)

            idx_list = [rr * w + cc for rr in range(r_min, r_max + 1) for cc in range(c_min, c_max + 1)]

            n = len(idx_list)
            pos_idx[j, :n] = torch.tensor(idx_list, dtype=torch.long, device=device)
            pos_mask[j, :n] = True

        return pos_idx, pos_mask
