# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn, optim

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .loss import FocalLoss
from .perlin import PerlinNoise
from .torch_model import GlassModel


class Glass(AnomalibModule):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        anomaly_source_path: str,
        backbone: str | nn.Module = "resnet18",
        pretrain_embed_dim: int = 1024,
        target_embed_dim: int = 1024,
        patchsize: int = 3,
        patchstride: int = 1,
        pre_trained: bool = True,
        layers: list[str] = ["layer1", "layer2", "layer3"],
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
    ):
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.perlin = PerlinNoise(anomaly_source_path)

        self.model = GlassModel(
            input_shape=input_shape,
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
        )

        self.c = torch.tensor([1])
        self.p = p
        self.radius = radius
        self.mining = mining
        self.noise = noise
        self.distribution = 0
        self.lr = lr
        self.step = step

        self.focal_loss = FocalLoss()

    def configure_optimizers(self) -> list[optim.Optimizer]:
        optimizers = []
        if not self.model.pre_trained:
            backbone_opt = optim.AdamW(self.model.foward_modules["feature_aggregator"].backbone.parameters(), self.lr)
            optimizers.append(backbone_opt)
        else:
            optimizers.append(None)

        if self.model.pre_proj > 0:
            proj_opt = optim.AdamW(self.model.pre_projection.parameters(), self.lr, weight_decay=1e-5)
            optimizers.append(proj_opt)
        else:
            optimizers.append(None)

        dsc_opt = optim.AdamW(self.model.discriminator.parameters(), lr=self.lr * 2)
        optimizers.append(dsc_opt)

        return optimizers

    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> STEP_OUTPUT:
        backbone_opt, proj_opt, dsc_opt = self.optimizers()

        self.model.forward_modules.eval()
        if self.model.pre_proj > 0:
            self.pre_projection.train()
        self.model.discriminator.train()

        dsc_opt.zero_grad()
        if proj_opt is not None:
            proj_opt.zero_grad()
        if backbone_opt is not None:
            backbone_opt.zero_grad()

        img = batch.image
        aug, mask_s = self.perlin(img)

        true_feats, fake_feats = self.model(img, aug)

        mask_s_gt = mask_s.reshape(-1, 1)
        noise = torch.normal(0, self.noise, true_feats.shape)
        gaus_feats = true_feats + noise

        center = self.c.repeat(img.shape[0], 1, 1)
        center = center.reshape(-1, center.shape[-1])
        true_points = torch.concat([fake_feats[mask_s_gt[:, 0] == 0], true_feats], dim=0)
        c_t_points = torch.concat([center[mask_s_gt[:, 0] == 0], center], dim=0)
        dist_t = torch.norm(true_points - c_t_points, dim=1)
        r_t = torch.tensor([torch.quantile(dist_t, q=self.radius)]).to(self.device)

        for step in range(self.step + 1):
            scores = self.model.discriminator(torch.cat([true_feats, gaus_feats]))
            true_scores = scores[: len(true_feats)]
            gaus_scores = scores[len(true_feats) :]
            true_loss = nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
            gaus_loss = nn.BCELoss()(gaus_scores, torch.ones_like(gaus_scores))
            bce_loss = true_loss + gaus_loss

            if step == self.step:
                break
            if self.mining == 0:
                dist_g = torch.norm(gaus_feats - center, dim=1)
                r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)])
                break

            grad = torch.autograd.grad(gaus_loss, [gaus_feats])[0]
            grad_norm = torch.norm(grad, dim=1)
            grad_norm = grad_norm.view(-1, 1)
            grad_normalized = grad / (grad_norm + 1e-10)

            with torch.no_grad():
                gaus_feats.add_(0.001 * grad_normalized)

        fake_points = fake_feats[mask_s_gt[:, 0] == 1]
        true_points = true_feats[mask_s_gt[:, 0] == 1]
        c_f_points = center[mask_s_gt[:, 0] == 1]
        dist_f = torch.norm(fake_points - c_f_points, dim=1)
        r_f = torch.tensor([torch.quantile(dist_f, q=self.radius)]).to(self.device)
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

        fake_scores = self.model.discriminator(fake_feats)

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

        loss = bce_loss + focal_loss
        loss.backward()

        if proj_opt is not None:
            proj_opt.step()
        if backbone_opt is not None:
            backbone_opt.step()
        dsc_opt.step()

    def on_train_start(self) -> None:
        dataloader = self.trainer.train_dataloader

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i == 0:
                    self.c = self.model.calculate_mean(batch.image)
                else:
                    self.c += self.model.calculate_mean(batch.image)

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
        """Return GLASS trainer arguments."""
        return {"gradient_clip_val": 0, "num_sanity_val_steps": 0}
        # TODO
