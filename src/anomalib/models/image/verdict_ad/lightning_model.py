# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""VerdictAD: few-shot anomaly detection via decision-guided refinement."""

from enum import Enum
from inspect import signature
from typing import Any, NoReturn

import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchvision.transforms.v2 import CenterCrop, Compose, InterpolationMode, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule, MemoryBankMixin
from anomalib.models.image.anomaly_dino.lightning_model import AnomalyDINO
from anomalib.models.image.dinomaly.components import StableAdamW, WarmCosineScheduler
from anomalib.models.image.dinomaly.lightning_model import TRAINING_CONFIG, Dinomaly
from anomalib.models.image.dinomaly.torch_model import DinomalyModel
from anomalib.models.image.patchcore.lightning_model import Patchcore
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .decider import validate_decision_options
from .torch_model import VerdictADModel, validate_refinement_options


class VerdictAD(MemoryBankMixin, AnomalibModule):
    """Refine AnomalyDINO, PatchCore, or Dinomaly predictions using visual decisions.

    The first ``n_shots`` normal training images supply Clef-Flash's references.
    Memory-bank detectors fit those images; Dinomaly trains its decoder and
    bottleneck on all normal training batches. Use one category per reference set.

    Args:
        n_shots (int): Number of normal references, one to four. Defaults to four.
        decision_model_path (str | None): Trusted local Clef-Flash release directory.
        decision_batch_size (int): Number of decision records per forward pass.
        alpha (float): Global refinement strength; zero disables GVR.
        gamma (float): Local refinement strength; zero disables LVR.
        pre_processor (PreProcessor | bool): Shared RGB preprocessing or enable flag.
        post_processor (PostProcessor | bool): Calibration/thresholding or enable flag.
        evaluator (Evaluator | bool): Evaluator or enable flag.
        visualizer (Visualizer | bool): Visualizer or enable flag.
        detector (AnomalyDINO | Patchcore | Dinomaly | dict[str, Any] | None): Configured
            detector instance, or its class_path/init_args configuration for CLI and
            checkpoint restoration. Defaults to AnomalyDINO with a DINOv2-L encoder.
            VerdictAD owns preprocessing, postprocessing, evaluation, and visualization.
        decision_prompts (dict[str, str] | None): Optional free/reference/crop/box question overrides.
    """

    def __init__(
        self,
        n_shots: int = 4,
        decision_model_path: str | None = None,
        decision_batch_size: int = 4,
        alpha: float = 0.5,
        gamma: float = 0.25,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
        detector: AnomalyDINO | Patchcore | Dinomaly | dict[str, Any] | None = None,
        decision_prompts: dict[str, str] | None = None,
    ) -> None:
        validate_refinement_options(n_shots, alpha, gamma)
        validate_decision_options(decision_batch_size, decision_prompts)
        supported = {f"anomalib.models.{model.__name__}": model for model in (AnomalyDINO, Patchcore, Dinomaly)}
        if isinstance(detector, dict):
            model_class = supported.get(detector.get("class_path", ""))
            if model_class is None or detector.keys() - {"class_path", "init_args"}:
                msg = "detector configuration must select AnomalyDINO, Patchcore, or Dinomaly."
                raise ValueError(msg)
            detector = model_class(**detector.get("init_args", {}))
        if detector is None:
            detector = AnomalyDINO(encoder_name="vit_large_patch14_dinov2")
        if type(detector) not in {AnomalyDINO, Patchcore, Dinomaly}:
            msg = "detector must be an AnomalyDINO, Patchcore, or Dinomaly instance."
            raise ValueError(msg)
        configured_detector = detector
        # Save constructor settings as plain data instead of pickling the live Lightning module.
        detector = None
        if isinstance(configured_detector, Dinomaly) and pre_processor is True:
            pre_processor = PreProcessor(Compose([Resize((448, 448)), CenterCrop(392)]))
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        self.save_hyperparameters({
            "detector": {
                "class_path": f"anomalib.models.{type(configured_detector).__name__}",
                "init_args": {
                    key: value.value if isinstance(value, Enum) else value
                    for key, value in configured_detector.hparams.items()
                    if key in signature(type(configured_detector).__init__).parameters
                },
            },
        })
        self._sampling_ratio = (
            configured_detector.coreset_sampling_ratio if isinstance(configured_detector, Patchcore) else 0.1
        )
        self._detector_trainer_arguments = {
            **configured_detector.trainer_arguments,
            "devices": 1,
            "num_sanity_val_steps": 0,
        }
        self.model = VerdictADModel(
            n_shots=n_shots,
            decision_model_path=decision_model_path,
            decision_batch_size=decision_batch_size,
            alpha=alpha,
            gamma=gamma,
            detector=configured_detector.model,
            decision_prompts=decision_prompts,
        )

    def training_step(self, batch: Batch, *args, **kwargs) -> torch.Tensor:
        """Collect references and compute Dinomaly's loss when selected.

        Args:
            batch (Batch): Normal training images.
            *args: Unused positional arguments.
            **kwargs: Unused keyword arguments.

        Returns:
            torch.Tensor: Dinomaly reconstruction loss or a dummy scalar loss.

        Raises:
            ValueError: If labeled anomalous training images are supplied.
        """
        del args, kwargs
        if batch.gt_label is not None and batch.gt_label.any():
            msg = "VerdictAD training references must all be normal."
            raise ValueError(msg)
        if batch.image is None:
            msg = "VerdictAD training requires an image batch."
            raise ValueError(msg)
        loss = self.model(batch.image, global_step=self.global_step)
        if isinstance(self.model.detector, DinomalyModel):
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.image.shape[0])
            return loss
        return torch.tensor(0.0, requires_grad=True, device=self.device)

    def fit(self) -> None:
        """Fit the detector on the collected references."""
        self.model.fit(sampling_ratio=self._sampling_ratio)

    def validation_step(self, batch: Batch, *args, **kwargs) -> Batch:
        """Refine predictions before calibration and evaluation.

        Args:
            batch (Batch): Query images and optional ground truth.
            *args: Unused positional arguments.
            **kwargs: Unused keyword arguments.

        Returns:
            Batch: Batch containing refined scores and maps.
        """
        del args, kwargs
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @classmethod
    def configure_pre_processor(cls, image_size: tuple[int, int] | int | None = None) -> PreProcessor:
        """Preserve RGB values for the decision model; normalize inside the detector.

        Args:
            image_size (tuple[int, int] | int | None): Target size, default ``(896, 896)``.

        Returns:
            PreProcessor: Bilinear resize without ImageNet normalization.
        """
        return PreProcessor(Resize(image_size or (896, 896), interpolation=InterpolationMode.BILINEAR, antialias=True))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Use Dinomaly's optimizer and schedule for its trainable decoder and bottleneck."""
        if isinstance(self.model.detector, DinomalyModel):
            parameters = list(self.model.detector.bottleneck.parameters()) + list(
                self.model.detector.decoder.parameters(),
            )
            optimizer = StableAdamW(parameters, **TRAINING_CONFIG["optimizer"])
            scheduler = WarmCosineScheduler(
                optimizer,
                **{**TRAINING_CONFIG["scheduler"], "total_iters": int(self.trainer.estimated_stepping_batches)},
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        return None

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return single-device settings appropriate to the selected detector."""
        return self._detector_trainer_arguments.copy()

    @property
    def learning_type(self) -> LearningType:
        """Return one-class learning with a bounded normal reference set."""
        return LearningType.ONE_CLASS

    def to_torch(self, *args, **kwargs) -> NoReturn:
        """Reject export; restore Lightning checkpoints with the external decision release instead."""
        del args, kwargs
        msg = f"{self.name} export is unsupported. Use a Lightning checkpoint and the local Clef-Flash release."
        raise NotImplementedError(msg)

    def to_onnx(self, *args, **kwargs) -> NoReturn:
        """Reject ONNX export of the PIL/custom-code decision pipeline."""
        self.to_torch(*args, **kwargs)

    def to_openvino(self, *args, **kwargs) -> NoReturn:
        """Reject OpenVINO export of the PIL/custom-code decision pipeline."""
        self.to_torch(*args, **kwargs)
