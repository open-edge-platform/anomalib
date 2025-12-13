# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model implementation for AnomalyDINO.

This module defines the low-level PyTorch implementation of the AnomalyDINO model,
which combines a DINOv2 Vision Transformer encoder with a memory-bank approach
for few-shot anomaly detection. It performs patch-based feature extraction,
optional background masking, and k-nearest neighbor search for anomaly scoring.

Example:
    >>> from anomalib.models.image.anomaly_dino.torch_model import AnomalyDINOModel
    >>> model = AnomalyDINOModel(
    ...     num_neighbours=1,
    ...     encoder_name="dinov2_vit_small_14",
    ...     masking=False,
    ...     coreset_subsampling=False,
    ...     sampling_ratio=0.1,
    ... )
"""

import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.data import InferenceBatch
from anomalib.models.components import DynamicBufferMixin, KCenterGreedy
from anomalib.models.components.dinov2 import DinoV2Loader
from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator


class AnomalyDINOModel(DynamicBufferMixin, nn.Module):
    """AnomalyDINO base PyTorch model for patch-based anomaly detection.

    This model uses DINOv2 transformers as feature extractors and applies
    a memory-bank mechanism for few-shot anomaly detection, similar to PatchCore.
    It supports optional background masking and coreset subsampling.

    Args:
        num_neighbours (int, optional): Number of nearest neighbors used for
            anomaly scoring. Defaults to ``1``.
        encoder_name (str, optional): DINOv2 encoder architecture name.
            Must start with ``"dinov2"``. Defaults to ``"dinov2_vit_small_14"``.
        masking (bool, optional): Whether to apply PCA-based masking to suppress
            background features. Defaults to ``False``.
        coreset_subsampling (bool, optional): Whether to apply greedy coreset
            selection to reduce memory bank size. Defaults to ``False``.
        sampling_ratio (float, optional): Fraction of samples retained during
            coreset subsampling. Defaults to ``0.1``.

    Example:
        >>> model = AnomalyDINOModel(masking=True, coreset_subsampling=True)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> preds = model(x)
        >>> preds.pred_score.shape
        torch.Size([1, 1])
    """

    def __init__(
        self,
        num_neighbours: int = 1,
        encoder_name: str = "dinov2_vit_small_14",
        masking: bool = False,
        coreset_subsampling: bool = False,
        sampling_ratio: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_neighbours = num_neighbours
        self.encoder_name = encoder_name
        self.masking = masking
        self.coreset_subsampling = coreset_subsampling
        self.sampling_ratio = sampling_ratio

        # Load DINOv2 backbone
        if not encoder_name.startswith("dinov2"):
            err_str = f"Encoder name must start with 'dinov2', got '{encoder_name}'"
            raise ValueError(err_str)
        self.feature_encoder = DinoV2Loader.from_name(self.encoder_name)
        self.feature_encoder.eval()

        # Memory bank and embedding storage
        self.register_buffer("memory_bank", torch.empty(0))
        self.embedding_store: list[torch.Tensor] = []

        # Anomaly map generator for visualization and scoring
        self.anomaly_map_generator = AnomalyMapGenerator()

    def fit(self) -> None:
        """Finalize and optionally subsample the memory bank after training.

        Once all embeddings from normal training images have been collected,
        this method consolidates them into the memory bank and optionally
        performs coreset-based subsampling.

        Raises:
            ValueError: If called before collecting any embeddings.
        """
        if len(self.embedding_store) == 0:
            err_str = "No embeddings collected. Run model in training mode first."
            raise ValueError(err_str)

        # Stack and normalize embeddings
        self.memory_bank = torch.vstack(self.embedding_store)
        self.embedding_store.clear()

        # Optional coreset selection
        if self.coreset_subsampling:
            sampler = KCenterGreedy(embedding=self.memory_bank, sampling_ratio=self.sampling_ratio)
            self.memory_bank = sampler.sample_coreset()

    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract patch-level feature embeddings from the last transformer layer.

        Returns flattened patch tokens excluding CLS and register tokens.

        Args:
            image_tensor (torch.Tensor): Input image tensor of shape ``(B, 3, H, W)``.

        Returns:
            torch.Tensor: Patch feature embeddings of shape ``(B, N, D)``,
            where ``N`` is the number of patches and ``D`` the feature dimension.
        """
        with torch.inference_mode():
            return self.feature_encoder.get_intermediate_layers(image_tensor, n=1)[0]

    @staticmethod
    def compute_background_masks(
        batch_features: np.ndarray,
        grid_size: tuple[int, int],
        threshold: float = 10.0,
        kernel_size: int = 3,
        border: float = 0.2,
    ) -> np.ndarray:
        """Compute binary masks to identify foreground patches.

        This method uses PCA on patch embeddings to estimate foreground regions,
        followed by morphological operations to clean up the mask.

        Args:
            batch_features (np.ndarray): Patch embeddings of shape ``(B, N, D)``.
            grid_size (tuple[int, int]): Spatial grid dimensions (H, W).
            threshold (float, optional): PCA threshold for foreground separation.
                Defaults to ``10.0``.
            kernel_size (int, optional): Morphological kernel size. Defaults to ``3``.
            border (float, optional): Fraction of image borders excluded from
                thresholding. Defaults to ``0.2``.

        Returns:
            np.ndarray: Boolean masks of shape ``(B, N)``, where ``True`` indicates
            foreground patches.
        """
        b, n, _ = batch_features.shape
        masks = np.ones((b, n), dtype=bool)

        for i in range(b):
            img_features = batch_features[i]
            pca = PCA(n_components=1, svd_solver="randomized")
            first_pc = pca.fit_transform(img_features.astype(np.float32))
            mask = first_pc > threshold

            mask_2d = mask.reshape(grid_size)
            h, w = grid_size
            y0, y1 = int(h * border), int(h * (1 - border))
            x0, x1 = int(w * border), int(w * (1 - border))
            center_crop = mask_2d[y0:y1, x0:x1]

            # Flip sign if PCA direction is inverted
            if center_crop.sum() <= center_crop.size * 0.35:
                mask = (-first_pc) > threshold
                mask_2d = mask.reshape(grid_size)

            # Morphological cleanup
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_2d = cv2.dilate(mask_2d.astype(np.uint8), kernel).astype(bool)
            mask_2d = cv2.morphologyEx(mask_2d.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

            masks[i] = mask_2d.flatten()

        return masks

    @staticmethod
    def mean_top1p(distances: torch.Tensor) -> torch.Tensor:
        """Compute the mean of the top 1% distances per image.

        Used as a robust aggregation of patch-level anomaly scores into a
        single image-level anomaly score.

        Args:
            distances (torch.Tensor): Patch-level distances of shape ``(B, N)``.

        Returns:
            torch.Tensor: Mean of the top 1% distances per image, shape ``(B, 1)``.
        """
        n = distances.shape[-1]
        num_top = max(int(n * 0.01), 1)
        topk_vals, _ = torch.topk(distances, num_top, dim=1, largest=True)
        return topk_vals.mean(dim=1, keepdim=True)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Forward pass for both training and inference.

        In training mode:
            - Extracts normalized patch features.
            - Collects embeddings into the memory bank.

        In inference mode:
            - Computes distances between input features and the memory bank.
            - Performs kNN-based scoring and anomaly map generation.

        Args:
            input_tensor (torch.Tensor): Input batch of shape ``(B, 3, H, W)``.

        Returns:
            Union[torch.Tensor, InferenceBatch]:
                - In training: dummy scalar tensor (no loss backprop).
                - In inference: :class:`anomalib.data.InferenceBatch` containing:
                    * ``pred_score``: Image-level anomaly score ``(B, 1)``
                    * ``anomaly_map``: Pixel-level anomaly heatmap ``(B, 1, H, W)``
        """
        # set precision
        input_tensor = input_tensor.type(self.memory_bank.dtype)

        # work out sizing
        b, _, h, w = input_tensor.shape
        cropped_width = w - w % self.feature_encoder.patch_size
        cropped_height = h - h % self.feature_encoder.patch_size
        grid_size = (
            cropped_height // self.feature_encoder.patch_size,
            cropped_width // self.feature_encoder.patch_size,
        )

        device = input_tensor.device
        features = self.extract_features(input_tensor)

        if self.masking:
            features_np = features.detach().cpu().numpy()
            masks_np = self.compute_background_masks(features_np, grid_size)
            masks = torch.from_numpy(masks_np).to(device)
        else:
            masks = torch.ones(features.shape[:2], dtype=torch.bool, device=device)

        features = features[masks]
        features = F.normalize(features, p=2, dim=1)

        if self.training:
            self.embedding_store.append(features)
            return torch.tensor(0.0, device=device, requires_grad=True)

        # check bank isn't empty at inference
        if self.memory_bank.numel() == 0:
            msg = "Memory bank is empty. Run the model in training mode and call `fit()` before inference."
            raise RuntimeError(msg)

        # Ensure dtype consistency
        if features.dtype != self.memory_bank.dtype:
            features = features.to(self.memory_bank.dtype)

        # Inference
        # L2-normalized distances
        # memory_bank : [M, D], features : [Q, D]

        # Compute cosine distance using matrix multiplication
        # both features and memory_bank are already L2-normalized.
        # cdist is not for half precision, but matmul is.
        similarity = torch.matmul(features, self.memory_bank.T)  # [Q, M]
        dists = (torch.ones_like(similarity) - similarity).clamp(min=0.0, max=2.0)  # cosine distance ∈ [0, 2]

        # Get top-k nearest neighbors
        k = max(1, self.num_neighbours)
        topk_vals, _ = torch.topk(dists, k=k, dim=1, largest=False)

        # Mean over k neighbors if needed
        min_dists = topk_vals.mean(dim=1) if k > 1 else topk_vals.squeeze(1)

        # Vectorized reconstruction
        distances_full = torch.zeros(
            (b, grid_size[0] * grid_size[1]),
            device=device,
            dtype=min_dists.dtype,
        )
        batch_idx, patch_idx = torch.nonzero(masks, as_tuple=True)
        distances_full[batch_idx, patch_idx] = min_dists

        # Aggregate image-level anomaly scores
        image_score = self.mean_top1p(distances_full)

        # Generate final anomaly map
        anomaly_map = distances_full.view(b, 1, *grid_size)
        anomaly_map = self.anomaly_map_generator(anomaly_map, (h, w))

        return InferenceBatch(pred_score=image_score, anomaly_map=anomaly_map)
