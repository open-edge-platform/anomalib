# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AnomalyDINO, PatchCore, and Dinomaly with VerdictAD's decision-guided refinement."""

from collections.abc import Callable
from typing import Literal

import torch
from torch import nn
from torchvision.transforms.v2 import functional as F  # noqa: N812

from anomalib.data import InferenceBatch
from anomalib.models.components import DynamicBufferMixin
from anomalib.models.image.anomaly_dino.torch_model import AnomalyDINOModel
from anomalib.models.image.dinomaly.torch_model import DinomalyModel
from anomalib.models.image.patchcore.torch_model import PatchcoreModel

from .decider import BOXES, ClefFlash
from .refinement import refine_predictions


class VerdictADModel(DynamicBufferMixin, nn.Module):
    """Fit a few-shot detector and refine its outputs with frozen Clef-Flash.

    Args:
        n_shots (int): Number of normal references, one to four.
        encoder_name (str): AnomalyDINO backbone name.
        decision_model_path (str | None): Trusted local Clef-Flash release directory.
        decision_batch_size (int): Decision records per Clef-Flash forward pass.
        alpha (float): Global refinement exponent.
        gamma (float): Local refinement exponent.
        detector (Literal["anomaly_dino", "patchcore", "dinomaly"]): Base detector.
        patchcore_backbone (str): PatchCore backbone.
        patchcore_layers (tuple[str, ...]): PatchCore feature layers.
        patchcore_pre_trained (bool): Load pretrained PatchCore weights.
        patchcore_coreset_sampling_ratio (float): Fraction of PatchCore embeddings retained.
        patchcore_num_neighbors (int): Neighbors used for PatchCore image scores.
        dinomaly_encoder_name (str): Dinomaly encoder backbone.
        decision_prompts (dict[str, str] | None): Optional free/reference/crop/box question overrides.
    """

    def __init__(
        self,
        n_shots: int = 4,
        encoder_name: str = "vit_large_patch14_dinov2",
        decision_model_path: str | None = None,
        decision_batch_size: int = 4,
        alpha: float = 0.5,
        gamma: float = 0.25,
        detector: Literal["anomaly_dino", "patchcore", "dinomaly"] = "anomaly_dino",
        patchcore_backbone: str = "wide_resnet50_2",
        patchcore_layers: tuple[str, ...] = ("layer2", "layer3"),
        patchcore_pre_trained: bool = True,
        patchcore_coreset_sampling_ratio: float = 0.1,
        patchcore_num_neighbors: int = 9,
        dinomaly_encoder_name: str = "vit_base_patch14_reg4_dinov2",
        decision_prompts: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(n_shots, int) or isinstance(n_shots, bool) or not 1 <= n_shots <= 4:
            msg = "n_shots must be between one and four."
            raise ValueError(msg)
        if not torch.isfinite(torch.tensor([alpha, gamma])).all() or min(alpha, gamma) < 0:
            msg = "alpha and gamma must be finite and nonnegative."
            raise ValueError(msg)
        if detector not in {"anomaly_dino", "patchcore", "dinomaly"}:
            msg = "detector must be 'anomaly_dino', 'patchcore', or 'dinomaly'."
            raise ValueError(msg)
        if detector == "patchcore" and not 0 < patchcore_coreset_sampling_ratio <= 1:
            msg = "patchcore_coreset_sampling_ratio must be in (0, 1]."
            raise ValueError(msg)
        if detector == "patchcore" and (
            not isinstance(patchcore_num_neighbors, int)
            or isinstance(patchcore_num_neighbors, bool)
            or patchcore_num_neighbors < 1
        ):
            msg = "patchcore_num_neighbors must be a positive integer."
            raise ValueError(msg)
        self.n_shots, self.alpha, self.gamma = n_shots, alpha, gamma
        self.decider = ClefFlash(decision_model_path, decision_batch_size, decision_prompts)
        self.detector: AnomalyDINOModel | PatchcoreModel | DinomalyModel
        if detector == "anomaly_dino":
            self.detector = AnomalyDINOModel(encoder_name=encoder_name)
        elif detector == "patchcore":
            self.detector = PatchcoreModel(
                backbone=patchcore_backbone,
                layers=patchcore_layers,
                pre_trained=patchcore_pre_trained,
                num_neighbors=patchcore_num_neighbors,
            )
        else:
            self.detector = DinomalyModel(encoder_name=dinomaly_encoder_name)
            self.detector.encoder.requires_grad_(requires_grad=False)
        self.patchcore_coreset_sampling_ratio = patchcore_coreset_sampling_ratio
        self.patchcore_num_neighbors = patchcore_num_neighbors
        self.register_buffer("reference_images", torch.empty(0))
        self.reference_images: torch.Tensor

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> "VerdictADModel":
        """Propagate device and dtype changes to weights kept outside checkpoints."""
        super()._apply(fn, recurse=recurse)
        if recurse and self.decider._model is not None:  # noqa: SLF001
            self.decider._model._apply(fn)  # noqa: SLF001
        return self

    def forward(self, images: torch.Tensor, global_step: int = 0) -> torch.Tensor | InferenceBatch:
        """Collect references and train Dinomaly; refine predictions in evaluation.

        Args:
            images (torch.Tensor): RGB images ``(B, 3, H, W)`` in ``[0, 1]``.
            global_step (int): Dinomaly's progressive loss schedule step.

        Returns:
            torch.Tensor | InferenceBatch: Dummy training loss or refined predictions.

        Raises:
            ValueError: If images are not finite RGB floats in ``[0, 1]``.
            RuntimeError: If evaluation is requested before fitting.
        """
        if images.ndim != 4 or images.shape[1] != 3 or images.numel() == 0:
            msg = "Expected a nonempty RGB image batch (B, 3, H, W)."
            raise ValueError(msg)
        # Bilinear resizing can overshoot [0, 1] by a few float32 ULPs.
        if (
            not images.is_floating_point()
            or not torch.isfinite(images).all()
            or images.min() < -1e-6
            or images.max() > 1 + 1e-6
        ):
            msg = "VerdictAD expects finite RGB floats in [0, 1]; do not apply external normalization."
            raise ValueError(msg)
        images = images.clamp(0, 1)
        if self.training:
            remaining = self.n_shots - len(self.reference_images)
            if remaining > 0:
                selected = images[:remaining].detach().clone()
                self.reference_images = (
                    selected if not self.reference_images.numel() else torch.cat((self.reference_images, selected))
                )
            if isinstance(self.detector, DinomalyModel):
                self.detector.encoder.eval()
                return self.detector(F.normalize(images, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), global_step)
            return images.new_zeros(())
        return self._predict(images)

    @torch.no_grad()
    def _predict(self, images: torch.Tensor) -> InferenceBatch:
        """Refine a detector's inference outputs without constructing a gradient graph."""
        if not self.reference_images.numel():
            msg = "Collect normal references and call fit() before inference."
            raise RuntimeError(msg)
        self.detector.eval()
        if isinstance(self.detector, PatchcoreModel):
            self.detector.num_neighbors = 1 if len(self.detector.memory_bank) == 1 else self.patchcore_num_neighbors
        predictions = self.detector(F.normalize(images, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        predictions = predictions._replace(pred_score=predictions.pred_score.reshape(-1))
        if self.alpha == 0 and self.gamma == 0:
            return predictions
        # PIL's NumPy conversion cannot represent bfloat16.
        references = [F.to_pil_image(image) for image in self.reference_images.to(device="cpu", dtype=torch.float32)]
        decisions = torch.stack([
            self.decider.predict(
                F.to_pil_image(image),
                references,
                images.device,
                include_global=self.alpha > 0,
                include_local=self.gamma > 0,
            )
            for image in images.to(device="cpu", dtype=torch.float32)
        ])
        boxes = BOXES * 2 if self.gamma > 0 else []
        global_logits, window_logits = decisions.split([2 if self.alpha else 0, len(boxes)], dim=1)
        return refine_predictions(predictions, global_logits, window_logits, boxes, self.alpha, self.gamma)

    @torch.no_grad()
    def fit(self) -> None:
        """Fit the detector using exactly the normal references used for decisions.

        Raises:
            ValueError: If fewer than ``n_shots`` training images were supplied.
        """
        if len(self.reference_images) != self.n_shots:
            msg = f"Expected {self.n_shots} normal references, collected {len(self.reference_images)}."
            raise ValueError(msg)
        if isinstance(self.detector, DinomalyModel):
            return
        self.detector.train()
        if isinstance(self.detector, AnomalyDINOModel):
            self.detector.feature_encoder.eval()
        else:
            self.detector.feature_extractor.eval()
        self.detector(F.normalize(self.reference_images, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        if isinstance(self.detector, AnomalyDINOModel):
            self.detector.fit()
        else:
            num_embeddings = sum(len(embedding) for embedding in self.detector.embedding_store)
            if int(num_embeddings * self.patchcore_coreset_sampling_ratio) < 1:
                self.detector.embedding_store.clear()
                msg = "PatchCore coreset would be empty; increase patchcore_coreset_sampling_ratio or image size."
                raise ValueError(msg)
            self.detector.subsample_embedding(self.patchcore_coreset_sampling_ratio)
        self.detector.eval()
