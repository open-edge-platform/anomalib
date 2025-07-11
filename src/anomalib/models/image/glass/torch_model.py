"""GLASS - Unsupervised anomaly detection via Gradient Ascent for Industrial Anomaly detection and localization.

This module implements the GLASS model for unsupervised anomaly detection and localization. GLASS synthesizes both
global and local anomalies using Gaussian noise guided by gradient ascent to enhance weak defect detection in
industrial settings.

The model consists of:
    - A feature extractor and feature adaptor to obtain robust normal representations
    - A Global Anomaly Synthesis (GAS) module that perturbs features using Gaussian noise and gradient ascent with
      truncated projection
    - A Local Anomaly Synthesis (LAS) module that overlays augmented textures onto images using Perlin noise masks
    - A shared discriminator trained with features from normal, global, and local synthetic samples

Paper: `A Unified Anomaly Synthesis Strategy with Gradient Ascent for Industrial Anomaly Detection and Localization
<https://arxiv.org/pdf/2407.09359>`
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

import kornia.filters as kf
import numpy as np
import torch
import torch.nn.functional as f
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.data.utils.generators.perlin import PerlinAnomalyGenerator
from anomalib.models.components import TimmFeatureExtractor
from anomalib.models.components.feature_extractors import dryrun_find_featuremap_dims

from .loss import FocalLoss


def init_weight(m: nn.Module) -> None:
    """Initializes network weights using Xavier normal initialization.

    Applies Xavier initialization for linear layers and normal initialization
    for convolutional and batch normalization layers.
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    if isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)


def _deduce_dims(
    feature_extractor: TimmFeatureExtractor,
    input_size: tuple[int, int],
    layers: list[str],
) -> list[int | tuple[int, int]]:
    """Determines feature dimensions for each layer in the feature extractor.

    Args:
        feature_extractor: The backbone feature extractor
        input_size: Input image dimensions
        layers: List of layer names to extract features from
    """
    dimensions_mapping = dryrun_find_featuremap_dims(
        feature_extractor,
        input_size,
        layers,
    )

    return [dimensions_mapping[layer]["num_features"] for layer in layers]


class Preprocessing(torch.nn.Module):
    """Handles initial feature preprocessing across multiple input dimensions.

    Input: List of features from different backbone layers
    Output: Processed features with consistent dimensionality
    """

    def __init__(self, input_dims: list[int | tuple[int, int]], output_dim: int) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for _ in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Applies preprocessing modules to a list of input feature tensors.

        Args:
            features (list of torch.Tensor): List of feature maps from different
                layers of the backbone network. Each tensor can have a different shape.

        Returns:
            torch.Tensor: A single tensor with shape (B, N, D), where B is the batch size,
            N is the number of feature maps, and D is the output dimension (`output_dim`).
        """
        features_ = []
        for module, feature in zip(self.preprocessing_modules, features, strict=False):
            features_.append(module(feature))
        return torch.stack(features_, dim=1)


class MeanMapper(torch.nn.Module):
    """Maps input features to a fixed dimension using adaptive average pooling.

    Input: Variable-sized feature tensors
    Output: Fixed-size feature representations
    """

    def __init__(self, preprocessing_dim: int) -> None:
        super().__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Applies adaptive average pooling to reshape features to a fixed size.

        Args:
            features (torch.Tensor): Input tensor of shape (B, *) where * denotes
            any number of remaining dimensions. It is flattened before pooling.

        Returns:
            torch.Tensor: Output tensor of shape (B, D), where D is `preprocessing_dim`.
        """
        features = features.reshape(len(features), 1, -1)
        return f.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    """Aggregates and reshapes features to a target dimension.

    Input: Multi-dimensional feature tensors
    Output: Reshaped and pooled features of specified target dimension
    """

    def __init__(self, target_dim: int) -> None:
        super().__init__()
        self.target_dim = target_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns reshaped and average pooled features."""
        features = features.reshape(len(features), 1, -1)
        features = f.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class Projection(torch.nn.Module):
    """Multi-layer projection network for feature adaptation.

    Args:
        in_planes: Input feature dimension
        out_planes: Output feature dimension
        n_layers: Number of projection layers
        layer_type: Type of intermediate layers
    """

    def __init__(self, in_planes: int, out_planes: int | None = None, n_layers: int = 1, layer_type: int = 0) -> None:
        super().__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        in_ = None
        out = None
        for i in range(n_layers):
            in_ = in_planes if i == 0 else out
            out = out_planes
            self.layers.add_module(f"{i}fc", torch.nn.Linear(in_, out))
            if i < n_layers - 1 and layer_type > 1:
                self.layers.add_module(f"{i}relu", torch.nn.LeakyReLU(0.2))
        self.apply(init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the projection network to the input features.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_planes), where B is the batch size.

        Returns:
            torch.Tensor: Transformed tensor of shape (B, out_planes).
        """
        return self.layers(x)


class Discriminator(torch.nn.Module):
    """Discriminator network for anomaly detection.

    Args:
        in_planes: Input feature dimension
        n_layers: Number of layers
        hidden: Hidden layer dimensions
    """

    def __init__(self, in_planes: int, n_layers: int = 2, hidden: int | None = None) -> None:
        super().__init__()

        hidden_ = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers - 1):
            in_ = in_planes if i == 0 else hidden_
            hidden_ = int(hidden_ // 1.5) if hidden is None else hidden
            self.body.add_module(
                f"block{i + 1}",
                torch.nn.Sequential(
                    torch.nn.Linear(in_, hidden_),
                    torch.nn.BatchNorm1d(hidden_),
                    torch.nn.LeakyReLU(0.2),
                ),
            )
        self.tail = torch.nn.Sequential(
            torch.nn.Linear(hidden_, 1, bias=False),
            torch.nn.Sigmoid(),
        )
        self.apply(init_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the discriminator network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_planes), where B is the batch size.

        Returns:
            torch.Tensor: Output tensor of shape (B, 1) containing probability scores.
        """
        x = self.body(x)
        return self.tail(x)


class PatchMaker:
    """Handles patch-based processing of feature maps.

    This class provides utilities for converting feature maps into patches,
    reshaping patch scores back to original dimensions, and computing global
    anomaly scores from patch-wise predictions.

    Attributes:
        patchsize (int): Size of each patch (patchsize x patchsize).
        stride (int or None): Stride used for patch extraction. Defaults to patchsize if None.
        top_k (int): Number of top patch scores to consider. Used for score reduction.
    """

    def __init__(self, patchsize: int, top_k: int = 0, stride: int | None = None) -> None:
        self.patchsize = patchsize
        self.stride = stride if stride is not None else patchsize
        self.top_k = top_k

    def patchify(
        self,
        features: torch.Tensor,
        return_spatial_info: bool = False,
    ) -> tuple[torch.Tensor, list[int]] | torch.Tensor:
        """Converts a batch of feature maps into patches.

        Args:
            features (torch.Tensor): Input feature maps of shape (B, C, H, W).
            return_spatial_info (bool): If True, also returns spatial patch count. Default is False.

        Returns:
            torch.Tensor: Output tensor of shape (B, N, C, patchsize, patchsize), where N is number of patches.
            list[int], optional: Number of patches in (height, width) dimensions, only if return_spatial_info is True.
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize,
            stride=self.stride,
            padding=padding,
            dilation=1,
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2],
            self.patchsize,
            self.patchsize,
            -1,
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    @staticmethod
    def unpatch_scores(x: torch.Tensor, batchsize: int) -> torch.Tensor:
        """Reshapes patch scores back into per-batch format.

        Args:
            x (torch.Tensor): Input tensor of shape (B * N, ...).
            batchsize (int): Original batch size.

        Returns:
            torch.Tensor: Reshaped tensor of shape (B, N, ...).
        """
        return x.reshape(batchsize, -1, *x.shape[1:])

    @staticmethod
    def score(x: torch.Tensor) -> torch.Tensor:
        """Computes final anomaly scores from patch-wise predictions.

        Args:
            x (torch.Tensor): Patch scores of shape (B, N, 1).

        Returns:
            torch.Tensor: Final anomaly score per image, shape (B,).
        """
        x = x[:, :, 0]  # remove last dimension if singleton
        return torch.max(x, dim=1).values


class RescaleSegmentor:
    """A utility class for rescaling and smoothing patch-level anomaly scores to generate segmentation masks.

    Attributes:
        target_size (int): The spatial size (height and width) to which patch scores will be rescaled.
        smoothing (int): The standard deviation used for Gaussian smoothing.
    """

    def __init__(self, target_size: int = 288) -> None:
        """Initializes the RescaleSegmentor.

        Args:
            target_size (int, optional): The desired output size (height/width) of segmentation maps. Defaults to 288.
        """
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(
        self,
        patch_scores: np.ndarray | torch.Tensor,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Converts patch-level scores to smoothed segmentation masks.

        Args:
            patch_scores (np.ndarray | torch.Tensor): Patch-wise scores of shape [N, H, W].
            device (torch.device): Device on which to perform computation.

        Returns:
            List[torch.Tensor]: A list of segmentation masks, each of shape [H, W],
                                rescaled to `target_size` and smoothed.
        """
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)

            scores = patch_scores.to(device)
            scores = scores.unsqueeze(1)  # [N, 1, H, W]
            scores = f.interpolate(
                scores, size=self.target_size, mode="bilinear", align_corners=False,
            )
            patch_scores = scores.squeeze(1)  # [N, H, W]

        patch_stack = patch_scores.unsqueeze(1)  # [N, 1, H, W]
        smoothed_stack = kf.gaussian_blur2d(
            patch_stack,
            kernel_size=(5, 5),
            sigma=(self.smoothing, self.smoothing),
        )

        return [s.squeeze(0) for s in smoothed_stack]  # List of [H, W] tensors


class GlassModel(nn.Module):
    """PyTorch Implementation of the GLASS Model."""

    def __init__(
        self,
        input_shape: tuple[int, int],  # (H, W)
        anomaly_source_path: str,
        pretrain_embed_dim: int = 1024,
        target_embed_dim: int = 1024,
        backbone: str = "resnet18",
        patchsize: int = 3,
        patchstride: int = 1,
        pre_trained: bool = True,
        layers: list[str] | None = None,
        pre_proj: int = 1,
        dsc_layers: int = 2,
        dsc_hidden: int = 1024,
        dsc_margin: float = 0.5,
        mining: int = 1,
        noise: float = 0.015,
        radius: float = 0.75,
        p: float = 0.5,
        lr: float = 0.0001,
        step: int = 20,
        svd: int = 0,
    ) -> None:
        super().__init__()

        if layers is None:
            layers = ["layer1", "layer2", "layer3"]

        self.backbone = backbone
        self.layers = layers
        self.input_shape = input_shape
        self.pre_trained = pre_trained

        self.augmentor = PerlinAnomalyGenerator(anomaly_source_path)

        self.focal_loss = FocalLoss()

        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = TimmFeatureExtractor(
            backbone=self.backbone,
            layers=self.layers,
            pre_trained=self.pre_trained,
        )
        feature_dimensions = _deduce_dims(feature_aggregator, self.input_shape, layers)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = Preprocessing(feature_dimensions, pretrain_embed_dim)
        self.forward_modules["preprocessing"] = preprocessing
        self.target_embed_dimension = target_embed_dim
        preadapt_aggregator = Aggregator(target_dim=target_embed_dim)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(
                self.target_embed_dimension,
                self.target_embed_dimension,
                pre_proj,
            )

        self.dsc_layers = dsc_layers
        self.dsc_hidden = dsc_hidden
        self.dsc_margin = dsc_margin
        self.discriminator = Discriminator(
            self.target_embed_dimension,
            n_layers=self.dsc_layers,
            hidden=self.dsc_hidden,
        )

        self.p = p
        self.radius = radius
        self.mining = mining
        self.noise = noise
        self.distribution = 0
        self.lr = lr
        self.step = step
        self.svd = svd

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.anomaly_segmentor = RescaleSegmentor(target_size=input_shape[:])

    def calculate_mean(self, images: torch.Tensor) -> torch.Tensor:
        """Computes the mean feature embedding across a batch of images.

        This method performs a forward pass through the model to extract feature embeddings
        for a batch of input images, optionally passing them through a pre-projection module.
        It then reshapes the output and calculates the mean across the batch dimension.

        Args:
            images (torch.Tensor): Input image tensor of shape (B, C, H, W), where:
                - B is the batch size,
                - C is the number of channels,
                - H and W are height and width.

        Returns:
            torch.Tensor: Mean embedding tensor of shape (N, D), where:
                - N is the number of patches or tokens per image,
                - D is the feature dimension.
        """
        self.forward_modules.eval()
        with torch.no_grad():
            if self.pre_proj > 0:
                outputs = self.pre_projection(self.generate_embeddings(images)[0])
                outputs = outputs[0] if len(outputs) == 2 else outputs
            else:
                outputs = self._embed(images, evaluation=False)[0]

            outputs = outputs[0] if len(outputs) == 2 else outputs
            outputs = outputs.reshape(images.shape[0], -1, outputs.shape[-1])

            return torch.mean(outputs, dim=0)

    def calculate_features(self,
        img: torch.Tensor,
        aug: torch.Tensor,
        evaluation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return feature embeddings for the input and augmented images.

        Depending on whether a pre-projection module is used, this method optionally applies it to the

        Args:
            img (torch.Tensor): The original input image tensor.
            aug (torch.Tensor): The augmented image tensor.
            evaluation (bool, optional): Whether the model is in evaluation mode. Defaults to False.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the feature embeddings for the original
                image (`true_feats`) and the augmented image (`fake_feats`).
        """
        if self.pre_proj > 0:
            fake_feats = self.pre_projection(
                self.generate_embeddings(aug, evaluation=evaluation)[0],
            )
            fake_feats = fake_feats[0] if len(fake_feats) == 2 else fake_feats
            true_feats = self.pre_projection(
                self.generate_embeddings(img, evaluation=evaluation)[0],
            )
            true_feats = true_feats[0] if len(true_feats) == 2 else true_feats
        else:
            fake_feats = self.generate_embeddings(aug, evaluation=evaluation)[0]
            fake_feats.requires_grad = True
            true_feats = self.generate_embeddings(img, evaluation=evaluation)[0]
            true_feats.requires_grad = True

        return true_feats, fake_feats

    def generate_embeddings(
        self,
        images: torch.Tensor,
        evaluation: bool = False,
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
        """Generates patch-wise feature embeddings for a batch of input images.

        This method performs a forward pass through the model's feature extraction pipeline,
        processes selected intermediate layers, reshapes them into patches, aligns their spatial sizes,
        and passes them through preprocessing and aggregation modules.

        Args:
            images (torch.Tensor): Input images of shape (B, C, H, W), where:
                - B is the batch size,
                - C is the number of channels,
                - H and W are the image height and width.
            evaluation (bool, optional): Whether to run in evaluation mode (disabling gradients).
                Default is False.

        Returns:
            tuple[list[torch.Tensor], list[tuple[int, int]]]:
                - A list of patch-level feature tensors, each of shape (N, D, P, P),
                where N is the number of patches, D is the channel dimension, and P is patch size.
                - A list of (height, width) tuples indicating the number of patches in each spatial dimension
                for each corresponding feature level.
        """
        if not evaluation and not self.pre_trained:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images)
        else:
            self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers]
        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape  # noqa: N806
                features[i] = feat.reshape(
                    B,
                    int(math.sqrt(L)),
                    int(math.sqrt(L)),
                    C,
                ).permute(0, 3, 1, 2)

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        patch_features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(patch_features)):
            features_ = patch_features[i]
            patch_dims = patch_shapes[i]

            features_ = features_.reshape(
                features_.shape[0],
                patch_dims[0],
                patch_dims[1],
                *features_.shape[2:],
            )
            features_ = features_.permute(0, 3, 4, 5, 1, 2)
            perm_base_shape = features_.shape
            features_ = features_.reshape(-1, *features_.shape[-2:])
            features_ = f.interpolate(
                features_.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            features_ = features_.squeeze(1)
            features_ = features_.reshape(
                *perm_base_shape[:-2],
                ref_num_patches[0],
                ref_num_patches[1],
            )
            features_ = features_.permute(0, 4, 5, 1, 2, 3)
            features_ = features_.reshape(len(features_), -1, *features_.shape[-3:])
            patch_features[i] = features_

        patch_features = [x.reshape(-1, *x.shape[-3:]) for x in patch_features]
        patch_features = self.forward_modules["preprocessing"](patch_features)
        patch_features = self.forward_modules["preadapt_aggregator"](patch_features)

        return patch_features, patch_shapes

    def calculate_anomaly_scores(self, images: torch.Tensor) -> torch.Tensor:
        """Calculates anomaly scores and segmentation masks for a batch of input images.

        Args:
            images (torch.Tensor): Batch of input images of shape [B, C, H, W].

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]:
                - image_scores: Tensor of anomaly scores per image, shape [B].
                - masks: List of segmentation masks for each image, each of shape [H, W].
        """
        with torch.no_grad():
            patch_features, patch_shapes = self.generate_embeddings(images, evaluation=True)
            if self.pre_proj > 0:
                patch_features = self.pre_projection(patch_features)
                patch_features = patch_features[0] if len(patch_features) == 2 else patch_features

            patch_scores = image_scores = self.discriminator(patch_features)
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, images.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(images.shape[0], scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores, device=images.device)

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=images.shape[0])
            image_scores = self.patch_maker.score(image_scores)

            return image_scores, masks

    def forward(
        self,
        img: torch.Tensor,
        c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to compute patch-wise feature embeddings for original and augmented images.

        Depending on whether a pre-projection module is used, this method optionally applies it to the
        embeddings generated for both `img` and `aug`. If not, the embeddings are directly obtained and
        `requires_grad` is enabled for them, likely for gradient-based optimization or anomaly generation.
        """
        device = img.device
        aug, mask_s = self.augmentor(img)
        if img is not None:
            batch_size = img.shape[0]

        true_feats, fake_feats = self.calculate_features(img, aug)

        h_ratio = mask_s.shape[2] // int(math.sqrt(fake_feats.shape[0] // batch_size))
        w_ratio = mask_s.shape[3] // int(math.sqrt(fake_feats.shape[0] // batch_size))

        mask_s_resized = f.interpolate(
            mask_s.float(),
            size=(mask_s.shape[2] // h_ratio, mask_s.shape[3] // w_ratio),
            mode="nearest",
        )
        mask_s_gt = mask_s_resized.reshape(-1, 1)

        noise = torch.normal(0, self.noise, true_feats.shape).to(device)
        gaus_feats = true_feats + noise

        center = c.repeat(img.shape[0], 1, 1)
        center = center.reshape(-1, center.shape[-1])
        true_points = torch.concat(
            [fake_feats[mask_s_gt[:, 0] == 0], true_feats],
            dim=0,
        )
        c_t_points = torch.concat([center[mask_s_gt[:, 0] == 0], center], dim=0)
        dist_t = torch.norm(true_points - c_t_points, dim=1)
        r_t = torch.tensor([torch.quantile(dist_t, q=self.radius)]).to(device)

        for step in range(self.step + 1):
            scores = self.discriminator(torch.cat([true_feats, gaus_feats]))
            true_scores = scores[: len(true_feats)]
            gaus_scores = scores[len(true_feats) :]
            true_loss = nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
            gaus_loss = nn.BCELoss()(gaus_scores, torch.ones_like(gaus_scores))
            bce_loss = true_loss + gaus_loss

            if step == self.step:
                break

            if self.training:
                grad = torch.autograd.grad(gaus_loss, [gaus_feats])[0]
                grad_norm = torch.norm(grad, dim=1)
                grad_norm = grad_norm.view(-1, 1)
                grad_normalized = grad / (grad_norm + 1e-10)

                with torch.no_grad():
                    gaus_feats.add_(0.001 * grad_normalized)

                if (step + 1) % 5 == 0:
                    dist_g = torch.norm(gaus_feats - center, dim=1)
                    proj_feats = center if self.svd == 1 else true_feats
                    r = r_t if self.svd == 1 else 0.5

                    h = gaus_feats - proj_feats
                    h_norm = dist_g if self.svd == 1 else torch.norm(h, dim=1)
                    alpha = torch.clamp(h_norm, r, 2 * r)
                    proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
                    h = proj * h
                    gaus_feats = proj_feats + h

        fake_points = fake_feats[mask_s_gt[:, 0] == 1]
        true_points = true_feats[mask_s_gt[:, 0] == 1]
        c_f_points = center[mask_s_gt[:, 0] == 1]
        dist_f = torch.norm(fake_points - c_f_points, dim=1)
        proj_feats = c_f_points if self.svd == 1 else true_points
        r = r_t if self.svd == 1 else 1

        if self.svd == 1:
            h = fake_points - proj_feats
            h_norm = dist_f if self.svd == 1 else torch.norm(h, dim=1)
            alpha = torch.clamp(h_norm, 2 * r, 4 * r)
            proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
            h = proj * h
            fake_points = proj_feats + h
            fake_feats[mask_s_gt[:, 0] == 1] = fake_points

        fake_scores = self.discriminator(fake_feats)

        if self.p > 0:
            fake_dist = (fake_scores - mask_s_gt) ** 2
            d_hard = torch.quantile(fake_dist, q=self.p)
            fake_scores_ = fake_scores[fake_dist >= d_hard].unsqueeze(1)
            mask_ = mask_s_gt[fake_dist >= d_hard].unsqueeze(1)
        else:
            fake_scores_ = fake_scores
            mask_ = mask_s_gt
        output = torch.cat([1 - fake_scores_, fake_scores_], dim=1)
        focal_loss = self.focal_loss(output, mask_)

        if self.training:
            loss = bce_loss + focal_loss
            return true_loss, gaus_loss, bce_loss, focal_loss, loss

        anomaly_scores, masks = self.calculate_anomaly_scores(img)
        masks = torch.stack(masks)
        return InferenceBatch(pred_score=anomaly_scores, anomaly_map=masks)
