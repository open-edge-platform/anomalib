# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch model for the SuperADD model implementation.

This module implements a PatchCore-style anomaly detector built on a DINOv3
backbone. Multi-layer ViT token features are extracted over overlapping image
patches, a per-layer memory bank is built from normal training images via
distance-based coreset subsampling, and test images are scored by 1-nearest-
neighbor distance to the bank.

The model stores representative patch features from normal training images and
detects anomalies by comparing test image patches against this memory bank using
nearest neighbor search.

See Also:
    - :class:`anomalib.models.image.super_add.lightning_model.SuperADD`:
        Lightning implementation of the SuperADD model
    - :class:`anomalib.models.image.super_add.anomaly_map.AnomalyMapGenerator`:
        Anomaly map generation using nearest neighbor search
"""

import gc
import itertools
import logging
import math
from collections import defaultdict
from collections.abc import Sequence

import torch
from torch import nn
from tqdm import tqdm

from anomalib.data import InferenceBatch
from anomalib.models.components import DynamicBufferMixin

from .anomaly_map import AnomalyMapGenerator

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1024

import time
from contextlib import contextmanager


@contextmanager
def timer(name, sync=True):
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"{name}: {(time.perf_counter() - t0) * 1000:.2f} ms")


class DinoV3Backbone(nn.Module):
    def __init__(self, backbone: str, layers: list[int], weights_path: str):
        super().__init__()

        self.dino = torch.hub.load(
            "facebookresearch/dinov3",
            model=backbone,
            pretrained=True,
            weights=weights_path,
        ).eval()
        self.layers = layers
        self.model_patch_size = self.dino.patch_embed.patch_size[0]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        with torch.inference_mode():
            result = self.dino.get_intermediate_layers(x, n=self.layers, norm=False)
            return result


class PatchedExecution(nn.Module):
    def __init__(self, model: nn.Module, patch_size: int, patch_overlap: int, model_patch_size: int):
        super().__init__()
        assert patch_overlap > 0
        assert patch_size > 2 * patch_overlap
        assert patch_size % model_patch_size == 0
        assert patch_overlap % model_patch_size == 0

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.model_patch_size = model_patch_size
        self.model = model.half()

    def axis_patch_split(self, dim_size):
        assert dim_size >= self.patch_size

        dim_size_t = dim_size // self.model_patch_size
        overlap_t = self.patch_overlap // self.model_patch_size
        patch_size_t = self.patch_size // self.model_patch_size

        n_patches = math.ceil((dim_size_t - patch_size_t) / (patch_size_t - 2 * overlap_t)) + 1

        fac = dim_size_t - patch_size_t
        div = max(1, n_patches - 1)

        input_rois = []
        for i in range(n_patches):
            patch_min = (i * fac) // div
            patch_max = patch_min + patch_size_t
            input_rois.append((patch_min, patch_max))

        prediction_rois = []
        for i in range(n_patches):
            start = 0 if i == 0 else math.ceil((input_rois[i - 1][1] - input_rois[i][0]) / 2)
            end = (
                patch_size_t
                if i == n_patches - 1
                else patch_size_t - math.floor((input_rois[i][1] - input_rois[i + 1][0]) / 2)
            )
            prediction_rois.append((start, end))

        result_rois = [(0, prediction_rois[0][1] - prediction_rois[0][0])]
        for i in range(1, n_patches):
            start = result_rois[i - 1][1]
            end = start + prediction_rois[i][1] - prediction_rois[i][0]
            result_rois.append((start, end))

        input_rois = [(s * self.model_patch_size, e * self.model_patch_size) for (s, e) in input_rois]

        return input_rois, prediction_rois, result_rois

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 4

        b, c, h, w = x.shape
        patch_w, patch_h = self.patch_size, self.patch_size

        input_rois_y, prediction_rois_y, result_rois_y = self.axis_patch_split(h)
        input_rois_x, prediction_rois_x, result_rois_x = self.axis_patch_split(w)
        x_overlapped = torch.zeros(
            (b, len(input_rois_y) * len(input_rois_x), c, patch_h, patch_w),
            device=x.device,
            dtype=x.dtype,
        )

        for i, ((y_start, y_end), (x_start, x_end)) in enumerate(itertools.product(input_rois_y, input_rois_x)):
            x_overlapped[:, i] = x[:, :, y_start:y_end, x_start:x_end]

        x_overlapped = x_overlapped.reshape(-1, c, patch_h, patch_w)
        with torch.no_grad():
            prediction = self.model(x_overlapped)
        tokens_y, tokens_x = patch_h // self.model_patch_size, patch_w // self.model_patch_size

        prediction = torch.stack(prediction)
        vector_count, _, _, feature_count = prediction.shape

        prediction = prediction.reshape(vector_count, b, -1, tokens_y, tokens_x, feature_count)

        result = torch.zeros(
            (vector_count, b, h // self.model_patch_size, w // self.model_patch_size, feature_count),
            device=prediction.device,
            dtype=x.dtype,
        )

        prediction_rois = itertools.product(prediction_rois_y, prediction_rois_x)
        result_rois = itertools.product(result_rois_y, result_rois_x)
        for i, ((pred_roi_y, pred_roi_x), (res_roi_y, res_roi_x)) in enumerate(
            zip(prediction_rois, result_rois, strict=False),
        ):
            p = prediction[:, :, i, pred_roi_y[0] : pred_roi_y[1], pred_roi_x[0] : pred_roi_x[1]]
            result[:, :, res_roi_y[0] : res_roi_y[1], res_roi_x[0] : res_roi_x[1]] = p

        # result = [t.cpu().numpy() for t in result]
        result = [t for t in result]

        return result


class SuperADDModel(DynamicBufferMixin, nn.Module):
    """ """

    def __init__(
        self,
        layers: Sequence[str] = [3, 5, 7, 10],
        backbone: str = "dinov3_vits16",
        weights_path: str = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        patch_size: int = 448,
        patch_overlap: int = 16,
        max_database_size: int = 100000,
        subsampling_iterations: int = 100,
        patch_batch_size: int = 8,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.layers = layers
        self.backbone = DinoV3Backbone(backbone, layers, weights_path)
        self.patch_exec = PatchedExecution(self.backbone, patch_size, patch_overlap, self.backbone.model_patch_size)
        self.max_database_size = max_database_size
        self.subsampling_iterations = subsampling_iterations

        self.threshold = 0

        self.anomaly_map_generator = AnomalyMapGenerator()

        self.embedding_store: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.memory_bank: torch.Tensor

        self.register_buffer("memory_bank", torch.empty(0))

    def __clear_cache(self):
        torch.cuda.empty_cache()
        gc.collect()

    def _compute_device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    def subsample_embedding(self) -> None:
        logger.info("Subsampling embeddings ...")
        device = self._compute_device()
        for layer_idx, layer in enumerate(self.layers):
            # Embeddings live on CPU (offloaded during training); concatenate and
            # subsample there, then move the (small) coreset to the compute device.
            embeddings = torch.cat(self.embedding_store[layer_idx], dim=0)

            embeddings = self.subsampling_distance_based_fast(
                embeddings,
                self.max_database_size,
                iterations=self.subsampling_iterations,
                normalize=False,
                knn_neighbors=100,
            )
            embeddings = embeddings.to(device)
            if self.memory_bank.numel() == 0:
                self.memory_bank = torch.empty(
                    len(self.layers),
                    embeddings.shape[0],
                    embeddings.shape[1],
                    dtype=embeddings.dtype,
                    device=device,
                )

            self.memory_bank[layer_idx] = embeddings
            self.embedding_store[layer_idx] = []
            del embeddings
        del self.embedding_store
        self.__clear_cache()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """ """
        with timer("full pass"):
            input_tensor = input_tensor.type(self.memory_bank.dtype)
            if self.training:
                prediction = self.patch_exec(input_tensor)
                for layer_idx, (_, embedding) in enumerate(zip(self.layers, prediction, strict=False)):
                    embedding = embedding.reshape(-1, embedding.shape[-1])
                    self.embedding_store[layer_idx].append(embedding.to("cpu", non_blocking=True))

                return embedding  # Return the embedding for the training step (not used for loss computation)

            input_shape = input_tensor.shape
            with timer("model run"):
                predicted_embeddings = self.patch_exec(input_tensor)

            layer_distances = []
            with timer("nn search full"):
                for layer_idx, (_, predicted_embedding) in enumerate(
                    zip(self.layers, predicted_embeddings, strict=False),
                ):
                    b, h, w, c = predicted_embedding.shape
                    query = predicted_embedding.reshape(b, h * w, c)
                    keys = self.memory_bank[layer_idx]
                    # Chunk over the query (token) dimension so we never materialize the
                    # full [b, h*w, N_bank] distance matrix at once.
                    with timer(f"nn search layer {layer_idx}"):
                        dists, _ = self.nearest_neighbors_chunked(query, keys, knn_neighbors=1)
                    dists = dists.mean(dim=-1)
                    dists = (
                        dists.reshape(b, h, w) / c
                    )  # normalize distance by dimsize to account for different dimsizes
                    layer_distances.append(dists)
            with timer("finalize"):
                layer_distances = torch.stack(layer_distances, dim=1)
                layer_distances = torch.nn.functional.interpolate(
                    layer_distances,
                    size=(input_shape[-2], input_shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_map = torch.mean(layer_distances, dim=1)
                pred_score = anomaly_map.amax(dim=(1, 2))

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    def subsample(self, x, target_number_of_samples, knn_neighbors, normalize):
        device = self._compute_device()
        x = x.to(device)
        dists, _ = self.nearest_neighbors(x, x, knn_neighbors=knn_neighbors, normalize=normalize)
        # Start with a small distance and increase until we have fewer than the target number of samples
        target_distance_between_samples = dists.float().mean() / 10
        number_of_samples = target_number_of_samples + 1  # Initialize > target to enter the loop
        random_numbers = torch.rand(len(x), device=device)
        keep_mask = torch.zeros(len(x), dtype=torch.bool, device=device)

        while number_of_samples > target_number_of_samples:
            subsampling_factor = (dists < target_distance_between_samples).sum(dim=-1) + 1
            keep_mask = random_numbers < (1.0 / subsampling_factor.float())
            number_of_samples = int(keep_mask.sum())
            target_distance_between_samples *= 1.1  # Increase distance if still too many samples

        return keep_mask.cpu()

    def subsampling_distance_based_fast(
        self,
        features,
        target_number_of_samples,
        iterations=100,
        normalize=False,
        knn_neighbors=100,
    ):
        # perform subsample iteratively on random subsets of the data to speed up the nearest neighbor search
        keep_mask_total = torch.zeros(len(features), dtype=torch.bool)
        size_of_subsets = int(1 / iterations * len(features))  # Size of subsets
        target_to_keep_subset = target_number_of_samples // iterations  # Target to keep per subset

        for _ in tqdm(range(iterations), desc="Subsampling embeddings"):
            candidate_indices = torch.where(~keep_mask_total)[0]
            # Randomly select a subset of candidates for this iteration
            n = min(size_of_subsets, len(candidate_indices))
            perm = torch.randperm(len(candidate_indices))[:n]
            indices = candidate_indices[perm]

            keep_mask_subset = self.subsample(features[indices], target_to_keep_subset, knn_neighbors, normalize)

            keep_mask_total[indices] = keep_mask_subset  # Update total keep mask with this subset's results
            number_of_samples = int(keep_mask_total.sum())

        difference = target_number_of_samples - number_of_samples

        # Randomly select some unkept samples to add back until we reach the target number of samples
        if difference > 0:
            indices_to_add_back = torch.where(~keep_mask_total)[0]
            shuffle = torch.randperm(len(indices_to_add_back))
            indices_to_add_back = indices_to_add_back[shuffle]
            keep_mask_total[indices_to_add_back[:difference]] = True

        # do the subsampling
        features_subsampled = features[keep_mask_total]

        return features_subsampled

    def nearest_neighbors(self, features_query, features_key, knn_neighbors, normalize=False):
        if normalize:
            features_query = torch.nn.functional.normalize(features_query, dim=-1)
            features_key = torch.nn.functional.normalize(features_key, dim=-1)

        dists_full = torch.cdist(
            features_query,
            features_key,
            compute_mode="use_mm_for_euclid_dist",
        )  # [Nq, Nk] Fast GEMM-based distance computation

        topk_vals, topk_idx = torch.topk(
            dists_full,
            k=knn_neighbors,
            dim=-1,
            largest=False,  # Set largest=False to get smallest distances
            sorted=False,
        )
        return topk_vals, topk_idx

    def nearest_neighbors_chunked(
        self,
        features_query: torch.Tensor,
        features_key: torch.Tensor,
        knn_neighbors: int,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        normalize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked nearest-neighbor search over the query (token) dimension.

        Produces the same result as :meth:`nearest_neighbors` but processes the
        queries in chunks so the full ``[b, n_query, n_key]`` distance matrix is
        never materialized at once. Each distance block is reduced to its top-k
        and discarded, keeping peak memory bounded by the key-bank size rather
        than the number of query tokens.

        Args:
            features_query (torch.Tensor): Query features of shape ``(b, n_query, c)``.
            features_key (torch.Tensor): Key features of shape ``(n_key, c)``.
            knn_neighbors (int): Number of nearest neighbors to return.
            chunk_size (int): Max number of query tokens processed per block.
            normalize (bool): Whether to L2-normalize before the distance computation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(distances, indices)`` each of
                shape ``(b, n_query, knn_neighbors)``.
        """
        n_query = features_query.shape[1]
        if n_query <= chunk_size:
            return self.nearest_neighbors(
                features_query,
                features_key,
                knn_neighbors=knn_neighbors,
                normalize=normalize,
            )

        vals: list[torch.Tensor] = []
        idxs: list[torch.Tensor] = []
        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)
            chunk_vals, chunk_idx = self.nearest_neighbors(
                features_query[:, start:end],
                features_key,
                knn_neighbors=knn_neighbors,
                normalize=normalize,
            )
            vals.append(chunk_vals)
            idxs.append(chunk_idx)
        return torch.cat(vals, dim=1), torch.cat(idxs, dim=1)
