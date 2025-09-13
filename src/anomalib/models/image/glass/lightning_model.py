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

from pathlib import Path
from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import optim
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.data.utils import DownloadInfo, download_and_extract
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .torch_model import GlassModel

DTD_DOWNLOAD_INFO = DownloadInfo(
    name="dtd-r1.0.1.tar.gz",
    url="https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
    hashsum="e42855a52a4950a3b59612834602aa253914755c95b0cff9ead6d07395f8e205",
)


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
          Defaults to `(288, 288)`.
        anomaly_source_path (str): Path to the dataset or source directory containing normal images and anomaly textures
        backbone (str, optional): Name of the CNN backbone used for feature extraction.
            Defaults to `"wide_resnet50_2"`.
        pretrain_embed_dim (int, optional): Dimensionality of features extracted by the pre-trained backbone before
          adaptation.
            Defaults to `1536`.
        target_embed_dim (int, optional): Dimensionality of the target adapted features after projection.
            Defaults to `1536`.
        patchsize (int, optional): Size of the local patch used in feature aggregation (e.g., for neighborhood pooling).
            Defaults to `3`.
        patchstride (int, optional): Stride used when extracting patches for local feature aggregation.
            Defaults to `1`.
        pre_trained (bool, optional): Whether to use ImageNet pre-trained weights for the backbone network.
            Defaults to `True`.
        layers (list[str], optional): List of backbone layers to extract features from.
            Defaults to `["layer1", "layer2", "layer3"]`.
        pre_projection (int, optional): Number of projection layers used in the feature adaptor (e.g., MLP before
          discriminator).
            Defaults to `1`.
        discriminator_layers (int, optional): Number of layers in the discriminator network.
            Defaults to `2`.
        discriminator_hidden (int, optional): Number of hidden units in each discriminator layer.
            Defaults to `1024`.
        discriminator_margin (float, optional): Margin used for contrastive or binary classification loss in
          discriminator training.
            Defaults to `0.5`.
        learning_rate (float, optional): Learning rate for training the feature adaptor and discriminator networks.
            Defaults to `0.0001`.
        step (int, optional): Number of gradient ascent steps for anomaly synthesis.
            Defaults to `20`.
        svd (int, optional): Flag to enable SVD-based feature projection.
            Defaults to `0`.
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
    """

    def __init__(
        self,
        input_shape: tuple[int, int] = (288, 288),
        anomaly_source_path: str | None = None,
        backbone: str = "wide_resnet50_2",
        pretrain_embed_dim: int = 1536,
        target_embed_dim: int = 1536,
        patchsize: int = 3,
        patchstride: int = 1,
        pre_trained: bool = True,
        layers: list[str] | None = None,
        pre_projection: int = 1,
        discriminator_layers: int = 2,
        discriminator_hidden: int = 1024,
        discriminator_margin: float = 0.5,
        learning_rate: float = 0.0001,
        step: int = 20,
        svd: int = 0,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        if layers is None:
            layers = ["layer2", "layer3"]

        if anomaly_source_path is not None:
            dtd_dir = Path(anomaly_source_path)
            if not dtd_dir.is_dir():
                download_and_extract(dtd_dir, DTD_DOWNLOAD_INFO)

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
            pre_projection=pre_projection,
            discriminator_layers=discriminator_layers,
            discriminator_hidden=discriminator_hidden,
            discriminator_margin=discriminator_margin,
            step=step,
            svd=svd,
        )

        self.learning_rate = learning_rate
        self.pre_trained = pre_trained

        if pre_projection > 0:
            self.projection_opt = optim.Adam(
                self.model.projection.parameters(),
                self.learning_rate,
                weight_decay=1e-5,
            )
        else:
            self.projection_opt = None

        if not self.pre_trained:
            self.backbone_opt = optim.AdamW(
                self.model.forward_modules["feature_aggregator"].backbone.parameters(),
                self.learning_rate,
            )
        else:
            self.backbone_opt = None

        self.automatic_optimization = False

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
        image_size = image_size or (288, 288)

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
        return optim.AdamW(self.model.discriminator.parameters(), lr=self.learning_rate * 2)

    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        """Training step for GLASS model.

        Args:
            batch (Batch): Input batch containing images and metadata
            batch_idx (int): Index of the current batch

        Returns:
            STEP_OUTPUT: Dictionary containing loss values and metrics
        """
        del batch_idx

        discriminator_opt = self.optimizers()

        if not self.pre_trained:
            self.model.forward_modules["feature_aggregator"].train()
        if self.model.pre_projection > 0:
            self.model.projection.train()
        self.model.discriminator.train()

        discriminator_opt.zero_grad()
        if self.projection_opt is not None:
            self.projection_opt.zero_grad()
        if self.backbone_opt is not None:
            self.backbone_opt.zero_grad()

        true_loss, gaus_loss, bce_loss, focal_loss, loss = self.model(batch.image)
        self.manual_backward(loss)

        if self.projection_opt is not None:
            self.projection_opt.step()
        if self.backbone_opt is not None:
            self.backbone_opt.step()
        discriminator_opt.step()

        self.log("true_loss", true_loss, prog_bar=True)
        self.log("gaus_loss", gaus_loss, prog_bar=True)
        self.log("bce_loss", bce_loss, prog_bar=True)
        self.log("focal_loss", focal_loss, prog_bar=True)
        self.log("loss", loss, prog_bar=True)

    def validation_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        """Performs a single validation step during model evaluation.

        Args:
            batch (Batch): A batch of input data, typically containing images and ground truth labels.
            batch_idx (int): Index of the batch (unused in this function).

        Returns:
            STEP_OUTPUT: Output of the validation step, usually containing predictions and any associated metrics.
        """
        del batch_idx
        self.model.forward_modules.eval()

        if self.model.pre_projection > 0:
            self.model.projection.eval()
        self.model.discriminator.eval()

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    def on_train_epoch_start(self) -> None:
        """Initialize model by computing mean feature representation across training dataset.

        This method is called at the start of training and computes a mean feature vector
        that serves as a reference point for the normal class distribution.
        """
        dataloader = self.trainer.train_dataloader
        self.model.calculate_center(dataloader, self.device)

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
