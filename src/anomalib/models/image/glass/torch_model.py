# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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

import math

import torch
import torch.nn.functional as f
from torch import nn
from torch.utils.data import dataloader

from anomalib.data import InferenceBatch
from anomalib.data.utils.generators.perlin import PerlinAnomalyGenerator
from anomalib.models.components import TimmFeatureExtractor
from anomalib.models.components.feature_extractors import dryrun_find_featuremap_dims

from .components import Aggregator, Discriminator, PatchMaker, Preprocessing, Projection, RescaleSegmentor
from .loss import FocalLoss


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


class GlassModel(nn.Module):
    """PyTorch Implementation of the GLASS Model."""

    def __init__(
        self,
        input_shape: tuple[int, int] = (288, 288),  # (H, W)
        anomaly_source_path: str | None = None,
        pretrain_embed_dim: int = 1536,
        target_embed_dim: int = 1536,
        backbone: str = "wide_resnet50_2",
        patchsize: int = 3,
        patchstride: int = 1,
        pre_trained: bool = True,
        layers: list[str] | None = None,
        pre_projection: int = 1,
        discriminator_layers: int = 2,
        discriminator_hidden: int = 1024,
        discriminator_margin: float = 0.5,
        step: int = 20,
        svd: int = 0,
    ) -> None:
        super().__init__()

        if layers is None:
            layers = ["layer2", "layer3"]

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

        self.pre_projection = pre_projection
        if self.pre_projection > 0:
            self.projection = Projection(
                self.target_embed_dimension,
                self.target_embed_dimension,
                pre_projection,
            )

        self.discriminator_layers = discriminator_layers
        self.discriminator_hidden = discriminator_hidden
        self.discriminator_margin = discriminator_margin
        self.discriminator = Discriminator(
            self.target_embed_dimension,
            n_layers=self.discriminator_layers,
            hidden=self.discriminator_hidden,
        )

        self.distribution = 0
        self.step = step
        self.svd = svd

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.anomaly_segmentor = RescaleSegmentor(target_size=input_shape)

    def calculate_center(self, dataloader: dataloader, device: torch.device) -> None:
        """Calculates and updates the center embedding from a dataset.

        This method runs the model in evaluation mode and computes the mean feature
        representation (center) across the entire dataset. The center is used for
        further downstream tasks such as anomaly detection or feature normalization.

        Args:
            dataloader (DataLoader): A PyTorch DataLoader providing batches of data,
                                    where each batch contains an 'image' attribute.
            device (torch.device): The device on which tensors should be processed
                                (e.g., torch.device("cuda") or torch.device("cpu")).

        Returns:
            None: The method updates `self.center` in-place with the computed center tensor.
        """
        self.forward_modules.eval()
        self.center = torch.tensor([1])
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if self.pre_projection > 0:
                    outputs = self.projection(self.generate_embeddings(batch.image.to(device))[0])
                    outputs = outputs[0] if len(outputs) == 2 else outputs
                else:
                    outputs = self._embed(batch.image.to(device), evaluation=False)[0]

                outputs = outputs[0] if len(outputs) == 2 else outputs
                outputs = outputs.reshape(batch.image.to(device).shape[0], -1, outputs.shape[-1])

                if i == 0:
                    self.center = torch.mean(outputs, dim=0)
                else:
                    self.center += torch.mean(outputs, dim=0)

    def calculate_features(
        self,
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
        if self.pre_projection > 0:
            fake_feats = self.projection(
                self.generate_embeddings(aug, evaluation=evaluation)[0],
            )
            fake_feats = fake_feats[0] if len(fake_feats) == 2 else fake_feats
            true_feats = self.projection(
                self.generate_embeddings(img, evaluation=evaluation)[0],
            )
            true_feats = true_feats[0] if len(true_feats) == 2 else true_feats
        else:
            fake_feats = self.generate_embeddings(aug, evaluation=evaluation)[0]
            assert isinstance(fake_feats, torch.Tensor)
            fake_feats.requires_grad = True
            true_feats = self.generate_embeddings(img, evaluation=evaluation)[0]
            assert isinstance(true_feats, torch.Tensor)
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
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
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
            if self.pre_projection > 0:
                patch_features = self.projection(patch_features)
                patch_features = patch_features[0] if len(patch_features) == 2 else patch_features

            patch_scores = image_scores = self.discriminator(patch_features)
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, images.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(images.shape[0], scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores, device=images.device)

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=images.shape[0])
            image_scores = self.patch_maker.compute_score(image_scores)

            return image_scores, masks

    def forward(
        self,
        img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | InferenceBatch:
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

        noise = torch.normal(0, 0.015, true_feats.shape).to(device)
        gaus_feats = true_feats + noise

        center = self.center.repeat(img.shape[0], 1, 1)
        center = center.reshape(-1, center.shape[-1])
        true_points = torch.concat(
            [fake_feats[mask_s_gt[:, 0] == 0], true_feats],
            dim=0,
        )
        c_t_points = torch.concat([center[mask_s_gt[:, 0] == 0], center], dim=0)
        dist_t = torch.norm(true_points - c_t_points, dim=1)
        r_t = torch.tensor([torch.quantile(dist_t, q=0.75)]).to(device)

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

        fake_dist = (fake_scores - mask_s_gt) ** 2
        d_hard = torch.quantile(fake_dist, q=0.5)
        fake_scores_ = fake_scores[fake_dist >= d_hard].unsqueeze(1)
        mask_ = mask_s_gt[fake_dist >= d_hard].unsqueeze(1)

        output = torch.cat([1 - fake_scores_, fake_scores_], dim=1)
        focal_loss = self.focal_loss(output, mask_)

        if self.training:
            loss = bce_loss + focal_loss
            return true_loss, gaus_loss, bce_loss, focal_loss, loss

        anomaly_scores, masks = self.calculate_anomaly_scores(img)
        masks = torch.stack(masks)
        return InferenceBatch(pred_score=anomaly_scores, anomaly_map=masks)
