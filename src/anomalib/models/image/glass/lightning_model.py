import torch
from torch import nn
from torch import optim

from lightning.pytorch.utilities.types import STEP_OUTPUT

from anomalib.data import Batch
from anomalib.models.components import AnomalibModule
from anomalib.models.components import AnomalibModule
from anomalib.metrics import Evaluator
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .loss import FocalLoss
from .torch_model import GlassModel

class Glass(AnomalibModule):
    def __init__(
            self,
            backbone,
            input_shape,
            pretrain_embed_dim,
            target_embed_dim,
            patchsize: int = 3,
            patchstride: int = 1,
            pre_trained: bool = True,
            layers: list[str] = ["layer1", "layer2", "layer3"],
            pre_proj: int = 1,
            dsc_layers: int = 2,
            dsc_hidden: int = 1024,
            dsc_margin: int = 0.5,
            pre_processor: PreProcessor | bool = True,
            post_processor: PostProcessor | bool = True,
            evaluator: Evaluator | bool = True,
            visualizer: Visualizer | bool = True,
            mining: int = 1,
            noise: float = 0.015,
            radius: float = 0.75,
            p: float = 0.5,
            lr: int = 0.0001,
            step: int = 0
    ):
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

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
            dsc_margin=dsc_margin
        )

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
            batch_idx: int
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
        aug = batch.aug

        true_feats, fake_feats = self.model(img, aug)

        mask_s_gt = batch.mask_s.reshape(-1, 1)
        noise = torch.normal(0, self.noise, true_feats.shape)
        gaus_feats = true_feats + noise

        for step in range(self.step + 1):
            scores = self.model.discriminator(torch.cat([true_feats, gaus_feats]))
            true_scores = scores[:len(true_feats)]
            gaus_scores = scores[len(true_feats):]
            true_loss = nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
            gaus_loss = nn.BCELoss()(gaus_scores, torch.ones_like(gaus_scores))
            bce_loss = true_loss + gaus_loss

            if step == self.step:
                break

            grad = torch.autograd.grad(gaus_loss, [gaus_feats])[0]
            grad_norm = torch.norm(grad, dim=1)
            grad_norm = grad_norm.view(-1, 1)
            grad_normalized = grad / (grad_norm + 1e-10)

            with torch.no_grad():
                gaus_feats.add_(0.001 * grad_normalized)

        fake_scores = self.model.discriminator(fake_feats)

        if self.p > 0:
            fake_dist = (fake_scores - mask_s_gt) ** 2
            d_hard = torch.quantile(fake_dist, q=self.p)
            take_scores_ = fake_scores[fake_dist >= d_hard].unsqueeze(1)
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