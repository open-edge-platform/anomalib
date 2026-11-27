# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""AnomalyDINO, PatchCore, and Dinomaly with VerdictAD's decision-guided refinement."""

from collections.abc import Callable

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


def validate_refinement_options(n_shots: int, alpha: float, gamma: float) -> None:
    """Validate refinement settings before loading detector weights.

    Args:
        n_shots (int): Number of normal references, one to four.
        alpha (float): Global refinement exponent.
        gamma (float): Local refinement exponent.

    Raises:
        ValueError: If the reference count or refinement exponents are invalid.
    """
    if not isinstance(n_shots, int) or isinstance(n_shots, bool) or not 1 <= n_shots <= 4:
        msg = "n_shots must be between one and four."
        raise ValueError(msg)
    if not torch.isfinite(torch.tensor([alpha, gamma])).all() or min(alpha, gamma) < 0:
        msg = "alpha and gamma must be finite and nonnegative."
        raise ValueError(msg)


class VerdictADModel(DynamicBufferMixin, nn.Module):
    """Fit a few-shot detector and refine its outputs with frozen Clef-Flash.

    Args:
        n_shots (int): Number of normal references, one to four.
        decision_model_path (str | None): Trusted local Clef-Flash release directory.
        decision_batch_size (int): Decision records per Clef-Flash forward pass.
        alpha (float): Global refinement exponent.
        gamma (float): Local refinement exponent.
        detector (AnomalyDINOModel | PatchcoreModel | DinomalyModel | None): Configured
            PyTorch detector. Defaults to AnomalyDINO with a DINOv2-L encoder.
        decision_prompts (dict[str, str] | None): Optional free/reference/crop/box question overrides.
    """

    def __init__(
        self,
        n_shots: int = 4,
        decision_model_path: str | None = None,
        decision_batch_size: int = 4,
        alpha: float = 0.5,
        gamma: float = 0.25,
        detector: AnomalyDINOModel | PatchcoreModel | DinomalyModel | None = None,
        decision_prompts: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        validate_refinement_options(n_shots, alpha, gamma)
        if detector is not None and not isinstance(detector, (AnomalyDINOModel, PatchcoreModel, DinomalyModel)):
            msg = "detector must be an AnomalyDINOModel, PatchcoreModel, or DinomalyModel instance."
            raise ValueError(msg)
        self.n_shots, self.alpha, self.gamma = n_shots, alpha, gamma
        self.decider = ClefFlash(decision_model_path, decision_batch_size, decision_prompts)
        self.detector = detector if detector is not None else AnomalyDINOModel(encoder_name="vit_large_patch14_dinov2")
        if isinstance(self.detector, DinomalyModel):
            self.detector.encoder.requires_grad_(requires_grad=False)
        if isinstance(self.detector, PatchcoreModel) and (
            not isinstance(self.detector.num_neighbors, int)
            or isinstance(self.detector.num_neighbors, bool)
            or self.detector.num_neighbors < 1
        ):
            msg = "PatchCore num_neighbors must be a positive integer."
            raise ValueError(msg)
        self._patchcore_num_neighbors = self.detector.num_neighbors if isinstance(self.detector, PatchcoreModel) else 1
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
            self.detector.num_neighbors = 1 if len(self.detector.memory_bank) == 1 else self._patchcore_num_neighbors
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
    def fit(self, sampling_ratio: float = 0.1) -> None:
        """Fit the detector using exactly the normal references used for decisions.

        Args:
            sampling_ratio (float): Fraction of PatchCore embeddings retained in its coreset.

        Raises:
            ValueError: If references or the PatchCore sampling ratio are invalid.
        """
        if len(self.reference_images) != self.n_shots:
            msg = f"Expected {self.n_shots} normal references, collected {len(self.reference_images)}."
            raise ValueError(msg)
        if isinstance(self.detector, DinomalyModel):
            return
        if isinstance(self.detector, PatchcoreModel) and not 0 < sampling_ratio <= 1:
            msg = "PatchCore sampling_ratio must be in (0, 1]."
            raise ValueError(msg)
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
            if int(num_embeddings * sampling_ratio) < 1:
                self.detector.embedding_store.clear()
                msg = "PatchCore coreset would be empty; increase sampling_ratio or image size."
                raise ValueError(msg)
            self.detector.subsample_embedding(sampling_ratio)
        self.detector.eval()
