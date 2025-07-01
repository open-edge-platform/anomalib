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

from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.data.utils.generators.perlin import PerlinAnomalyGenerator
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import GlassModel


class Glass(AnomalibModule):
    """PyTorch Lightning Implementation of the GLASS Model.

    The model uses a pre-trained feature extractor to extract features and a feature adaptor to mitigate latent domain
      bias.
    Global anomaly features are synthesized from adapted normal features using gradient ascent.
    Local anomaly images are synthesized using texture overlay datasets like dtd which are then processed by feature
      extractor and feature adaptor.
    All three different features are passed to the discriminator trained using loss functions.

    Args:
        input_shape (tuple[int, int]): Input image dimensions as a tuple of (height, width). Required for shaping the
          input pipeline.
        anomaly_source_path (str): Path to the dataset or source directory containing normal images and anomaly textures
        backbone (str, optional): Name of the CNN backbone used for feature extraction.
            Defaults to `"resnet18"`.
        pretrain_embed_dim (int, optional): Dimensionality of features extracted by the pre-trained backbone before
          adaptation.
            Defaults to `1024`.
        target_embed_dim (int, optional): Dimensionality of the target adapted features after projection.
            Defaults to `1024`.
        patchsize (int, optional): Size of the local patch used in feature aggregation (e.g., for neighborhood pooling).
            Defaults to `3`.
        patchstride (int, optional): Stride used when extracting patches for local feature aggregation.
            Defaults to `1`.
        pre_trained (bool, optional): Whether to use ImageNet pre-trained weights for the backbone network.
            Defaults to `True`.
        layers (list[str], optional): List of backbone layers to extract features from.
            Defaults to `["layer1", "layer2", "layer3"]`.
        pre_proj (int, optional): Number of projection layers used in the feature adaptor (e.g., MLP before
          discriminator).
            Defaults to `1`.
        dsc_layers (int, optional): Number of layers in the discriminator network.
            Defaults to `2`.
        dsc_hidden (int, optional): Number of hidden units in each discriminator layer.
            Defaults to `1024`.
        dsc_margin (float, optional): Margin used for contrastive or binary classification loss in discriminator
          training.
            Defaults to `0.5`.
        pre_processor (PreProcessor | bool, optional): reprocessing module or flag to enable default preprocessing.
            Set to `True` to apply default normalization and resizing.
            Defaults to `True`.
        post_processor (PostProcessor | bool, optional): Postprocessing module or flag to enable default output
          smoothing or thresholding.
            Defaults to `True`.
        evaluator (Evaluator | bool, optional): Evaluation module for calculating metrics such as AUROC and PRO.
            Defaults to `True`.
        visualizer (Visualizer | bool, optional): Visualization module to generate heatmaps, segmentation overlays, and
          anomaly scores.
            Defaults to `True`.
        mining (int, optional): Number of iterations or difficulty level for Online Hard Example Mining (OHEM) during
          training.
            Defaults to `1`.
        noise (float, optional): Standard deviation of Gaussian noise used in feature-level anomaly synthesis.
            Defaults to `0.015`.
        radius (float, optional): Radius parameter used for truncated projection in the anomaly synthesis strategy.
            Determines the range for valid synthetic anomalies in the hypersphere or manifold.
            Defaults to `0.75`.
        p (float, optional): Probability used in random selection logic, such as anomaly mask generation or augmentation
          choice.
            Defaults to `0.5`.
        lr (float, optional): Learning rate for training the feature adaptor and discriminator networks.
            Defaults to `0.0001`.
        step (int, optional): Number of gradient ascent steps for anomaly synthesis.
            Defaults to `20`.
        svd (int, optional): Flag to enable SVD-based feature projection.
            Defaults to `0`.
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        anomaly_source_path: str,
        backbone: str = "resnet18",
        pretrain_embed_dim: int = 1024,
        target_embed_dim: int = 1024,
        patchsize: int = 3,
        patchstride: int = 1,
        pre_trained: bool = True,
        layers: list[str] | None = None,
        pre_proj: int = 1,
        dsc_layers: int = 2,
        dsc_hidden: int = 1024,
        dsc_margin: float = 0.5,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
        mining: int = 1,
        noise: float = 0.015,
        radius: float = 0.75,
        p: float = 0.5,
        lr: float = 0.0001,
        step: int = 20,
        svd: int = 0,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        if layers is None:
            layers = ["layer1", "layer2", "layer3"]

        self.augmentor = PerlinAnomalyGenerator(anomaly_source_path)

        self.model = GlassModel(
            input_shape=input_shape,
            anomaly_source_path=anomaly_source_path,
            pretrain_embed_dim=pretrain_embed_dim,
            target_embed_dim=target_embed_dim,
            backbone=backbone,
            pre_trained=pre_trained,
            patchsize=patchsize,
            patchstride=patchstride,
            layers=layers,
            pre_proj=pre_proj,
            dsc_layers=dsc_layers,
            dsc_hidden=dsc_hidden,
            dsc_margin=dsc_margin,
            step=step,
            svd=svd,
            mining=mining,
            noise=noise,
            radius=radius,
            p=p,
        )

        self.c = torch.tensor([1])
        self.lr = lr

        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if pre_proj > 0:
            self.proj_opt = optim.AdamW(
                self.model.pre_projection.parameters(),
                self.lr,
                weight_decay=1e-5,
            )
        else:
            self.proj_opt = None

        if not pre_trained:
            self.backbone_opt = optim.AdamW(
                self.mosdel.forward_modules["feature_aggregator"].backbone.parameters(),
                self.lr,
            )
        else:
            self.backbone_opt = None

    @classmethod
    def configure_pre_processor(
        cls,
        image_size: tuple[int, int] | None = None,
        center_crop_size: tuple[int, int] | None = None,
    ) -> PreProcessor:
        """Configure the default pre-processor for GLASS.

        If valid center_crop_size is provided, the pre-processor will
        also perform center cropping, according to the paper.

        Args:
            image_size (tuple[int, int] | None, optional): Target size for
                resizing. Defaults to ``(256, 256)``.
            center_crop_size (tuple[int, int] | None, optional): Size for center
                cropping. Defaults to ``None``.

        Returns:
            PreProcessor: Configured pre-processor instance.

        Raises:
            ValueError: If at least one dimension of ``center_crop_size`` is larger
                than correspondent ``image_size`` dimension.

        Example:
            >>> pre_processor = Glass.configure_pre_processor(
            ...     image_size=(256, 256)
            ... )
            >>> transformed_image = pre_processor(image)
        """
        image_size = image_size or (256, 256)

        if center_crop_size is not None:
            if center_crop_size[0] > image_size[0] or center_crop_size[1] > image_size[1]:
                msg = f"Center crop size {center_crop_size} cannot be larger than image size {image_size}."
                raise ValueError(msg)
            transform = Compose([
                Resize(image_size, antialias=True),
                CenterCrop(center_crop_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = Compose([
                Resize(image_size, antialias=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        return PreProcessor(transform=transform)

    def configure_optimizers(self) -> optim.Optimizer:
        """Configure optimizer for the discriminator.

        Returns:
            Optimizer: AdamW Optimizer for the discriminator.
        """
        return optim.AdamW(self.model.discriminator.parameters(), lr=self.lr * 2)

    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> STEP_OUTPUT:
        """Training step for GLASS model.

        Args:
            batch (Batch): Input batch containing images and metadata
            batch_idx (int): Index of the current batch

        Returns:
            STEP_OUTPUT: Dictionary containing loss values and metrics
        """
        del batch_idx
        dsc_opt = self.optimizers()

        self.model.forward_modules.eval()
        if self.model.pre_proj > 0:
            self.model.pre_projection.train()
        self.model.discriminator.train()

        dsc_opt.zero_grad()
        if self.proj_opt is not None:
            self.proj_opt.zero_grad()
        if self.backbone_opt is not None:
            self.backbone_opt.zero_grad()

        img = batch.image
        true_loss, gaus_loss, bce_loss, focal_loss, loss = self.model(img, self.c)
        loss.backward()

        if self.proj_opt is not None:
            self.proj_opt.step()
        if self.backbone_opt is not None:
            self.backbone_opt.step()
        dsc_opt.step()

        self.log("true_loss", true_loss, prog_bar=True)
        self.log("gaus_loss", gaus_loss, prog_bar=True)
        self.log("bce_loss", bce_loss, prog_bar=True)
        self.log("focal_loss", focal_loss, prog_bar=True)
        self.log("loss", loss, prog_bar=True)

    def on_train_start(self) -> None:
        """Initialize model by computing mean feature representation across training dataset.

        This method is called at the start of training and computes a mean feature vector
        that serves as a reference point for the normal class distribution.
        """
        dataloader = self.trainer.train_dataloader

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i == 0:
                    self.c = self.model.calculate_mean(batch.image.to(self.dev))
                else:
                    self.c += self.model.calculate_mean(batch.image.to(self.dev))

            self.c /= len(dataloader)

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type (ONE_CLASS for GLASS)
        """
        return LearningType.ONE_CLASS

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return GLASS trainer arguments.

        Returns:
            dict[str, Any]: Dictionary containing trainer configuration
        """
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}
