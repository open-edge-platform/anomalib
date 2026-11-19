# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""UniADet: language-free zero-/few-shot anomaly detection.

Paper: One Language-Free Foundation Model Is Enough for Universal Vision Anomaly Detection.
https://arxiv.org/abs/2601.05552

This is an independent implementation of the published algorithm, not a source
port of the unlicensed gaobb/UniADet repository. Auxiliary supervised pretraining
uses the standalone reproduction tool; Engine does not optimize the head.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torchvision.transforms.v2 import Compose, Normalize, Resize

from anomalib import LearningType, TaskType
from anomalib.data import AnomalibDataModule, ImageBatch
from anomalib.data.utils import read_image
from anomalib.data.utils.path import resolve_path_under_root
from anomalib.deploy import CompressionType
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer

from .backbones import BACKBONES
from .torch_model import UniADetModel

if TYPE_CHECKING:
    from lightning_utilities.core.imports import module_available
    from torchmetrics import Metric

    if module_available("openvino"):
        from openvino import Model as OVModel


class UniADet(AnomalibModule):
    """Zero-/few-shot detection with a separately pretrained auxiliary head.

    Use a separate model (and reference memory) per target category. Reference
    images are sampled only from normal training samples, never from test images.
    Full Lightning checkpoints preserve both the head and category memory.

    Args:
        backbone (str): One of ``"clip"``, ``"dinov2"`` or ``"dinov3"``.
        weights (str | Path | None): Auxiliary head checkpoint from ``tools/uniadet/train.py``.
        k_shot (int): Number of normal references; zero selects zero-shot detection.
        few_shot_source (str | Path | None): Directory of normal reference images.
            If omitted, use the attached datamodule's training split.
        seed (int): Seed used only for reference sampling.
        layers (tuple[int, ...]): One-based feature blocks. Defaults to the paper's five layers.
        pretrained (bool): Load pretrained encoder weights. Disable for offline checkpoint restoration.
        image_size (int | None): Square encoder resolution; defaults to 518 or 512.
        temperature (float): Cosine-logit temperature.
        few_shot_weight (float): Legacy reference-distance fusion weight; author protocol fixes this to 0.8.
        image_score_weight (float): Legacy image-map fusion weight; author protocol uses 0.1/0.5 for zero-/few-shot.
        pre_processor (nn.Module | bool): Input preprocessing component or enable flag.
        post_processor (nn.Module | bool): Prediction postprocessor or enable flag.
        evaluator (Evaluator | bool): Evaluation component or enable flag.
        visualizer (Visualizer | bool): Visualization component or enable flag.
        protocol (str): ``"legacy"`` preserves existing checkpoints; ``"author"`` selects
            the published executable's features, score processing and with-replacement references.
            Author heads must first be imported using ``tools.uniadet.author``.

    Raises:
        ValueError: If ``k_shot`` is negative or the backbone is unknown.

    Example:
        >>> from anomalib.models import UniADet
        >>> model = UniADet(weights="visa_head.pt", k_shot=4)  # doctest: +SKIP
        >>> # Engine uses normal target training images as references, without head optimization.
    """

    def __init__(
        self,
        backbone: str = "dinov2",
        weights: str | Path | None = None,
        k_shot: int = 0,
        few_shot_source: str | Path | None = None,
        seed: int = 10,
        layers: tuple[int, ...] = (12, 15, 18, 21, 24),
        pretrained: bool = True,
        image_size: int | None = None,
        temperature: float = 0.07,
        few_shot_weight: float = 0.5,
        image_score_weight: float = 0.5,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
        protocol: str = "legacy",
    ) -> None:
        if k_shot < 0:
            msg = "k_shot must be nonnegative."
            raise ValueError(msg)
        if backbone not in BACKBONES:
            msg = f"Unknown UniADet backbone {backbone!r}."
            raise ValueError(msg)
        self.backbone_name = backbone
        self.image_size = BACKBONES[backbone].image_size if image_size is None else image_size
        if pre_processor is True:
            pre_processor = self.configure_pre_processor((self.image_size, self.image_size), backbone)
        super().__init__(pre_processor, post_processor, evaluator, visualizer)
        self.k_shot = k_shot
        self.few_shot_source = Path(few_shot_source) if few_shot_source is not None else None
        self.seed = seed
        self.protocol = protocol
        self.model = UniADetModel(
            backbone,
            tuple(layers),
            pretrained,
            image_size,
            temperature,
            few_shot_weight,
            image_score_weight,
            protocol=protocol,
        )
        if weights is not None:
            self.model.load_head(weights)
        # A full checkpoint is self-contained and must not reopen an old head-file path.
        self.hparams["weights"] = None
        self.hparams["pretrained"] = False

    def _initialize_references(self) -> None:
        if not self.k_shot or self.model.memory_bank.shape[1]:
            return
        if self.few_shot_source is not None:
            extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            source = self.few_shot_source.resolve()
            paths = sorted(
                resolve_path_under_root(source, path) for path in source.rglob("*") if path.suffix.lower() in extensions
            )
        elif self._trainer is not None and self.trainer.datamodule is not None:
            dataset = self.trainer.datamodule.train_data
            normal = dataset.samples[dataset.samples.label_index == 0]
            paths = sorted(Path(path) for path in normal.image_path)
        else:
            msg = "Few-shot setup needs a normal reference directory or an attached training datamodule."
            raise ValueError(msg)
        if not paths or (self.protocol == "legacy" and len(paths) < self.k_shot):
            msg = f"Requested {self.k_shot} references but found only {len(paths)} normal images."
            raise ValueError(msg)
        generator = torch.Generator().manual_seed(self.seed)
        selected = (
            torch.randint(len(paths), (self.k_shot,), generator=generator).tolist()
            if self.protocol == "author"
            else torch.randperm(len(paths), generator=generator)[: self.k_shot].tolist()
        )
        images = []
        for index in selected:
            image = read_image(paths[index], as_tensor=True).to(self.device).unsqueeze(0)
            images.append(self.pre_processor(image) if self.pre_processor else image)
        self.model.set_reference_images(torch.cat(images))

    def on_validation_start(self) -> None:
        """Initialize normal reference memory after checkpoint restoration and device transfer."""
        self._initialize_references()

    def on_test_start(self) -> None:
        """Initialize reference memory before target evaluation."""
        self._initialize_references()

    def on_predict_start(self) -> None:
        """Initialize reference memory before Engine prediction."""
        self._initialize_references()

    def validation_step(self, batch: ImageBatch, batch_idx: int = 0) -> ImageBatch:
        """Predict a validation batch without updating the auxiliary head.

        Args:
            batch (ImageBatch): Preprocessed images and ground truth.
            batch_idx (int): Batch index (unused).

        Returns:
            ImageBatch: Input batch updated with predictions.
        """
        del batch_idx
        return batch.update(**self.model(batch.image)._asdict())

    @property
    def learning_type(self) -> LearningType:
        """Return the target-domain learning regime, never an optimizer-training workaround."""
        return LearningType.FEW_SHOT if self.k_shot else LearningType.ZERO_SHOT

    @property
    def trainer_arguments(self) -> dict[str, int | bool]:
        """Return no model-specific Trainer overrides."""
        return {}

    @staticmethod
    def configure_pre_processor(
        image_size: tuple[int, int] | None = None,
        backbone: str = "dinov2",
    ) -> PreProcessor:
        """Configure fixed-resolution preprocessing shared by Engine and deployment.

        Args:
            image_size (tuple[int, int] | None): Optional square resolution override.
            backbone (str): Encoder whose input size and normalization should be used.

        Returns:
            PreProcessor: Bilinear resizing followed by the encoder's normalization.
        """
        size = image_size or (BACKBONES[backbone].image_size,) * 2
        if backbone == "clip":
            mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        else:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        return PreProcessor(Compose([Resize(size, antialias=False), Normalize(mean, std)]))

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        """Keep raw paper scores while permitting threshold calibration."""
        return PostProcessor(enable_normalization=False)

    def _compress_ov_model(
        self,
        model: "OVModel",
        compression_type: CompressionType | None = None,
        datamodule: AnomalibDataModule | None = None,
        metric: "Metric | None" = None,
        task: TaskType | None = None,
        max_drop: float = 0.01,
    ) -> "OVModel":
        """Use transformer-aware quantization with floating-point map resizing and smoothing."""
        if compression_type not in {CompressionType.INT8_PTQ, CompressionType.INT8_ACQ}:
            return super()._compress_ov_model(model, compression_type, datamodule, metric, task, max_drop)
        import nncf

        # Auxiliary-domain ranges can clip unseen-domain anomaly maps when NNCF
        # quantizes fixed resizing or smoothing. Neither operation has learned weights.
        postprocessing = [
            node.friendly_name
            for node in model.get_ordered_ops()
            if node.get_type_name() == "Interpolate"
            or (
                node.get_type_name() == "Convolution"
                and node.input(1).get_partial_shape().is_static
                and list(node.input(1).get_shape())[:2] == [1, 1]
            )
        ]
        ignored_scope = nncf.IgnoredScope(names=postprocessing)
        if compression_type == CompressionType.INT8_PTQ:
            return self._post_training_quantization_ov(
                model,
                datamodule,
                ignored_scope=ignored_scope,
                model_type=nncf.ModelType.TRANSFORMER,
            )
        return self._accuracy_control_quantization_ov(
            model,
            datamodule,
            metric,
            task,
            max_drop,
            ignored_scope=ignored_scope,
            model_type=nncf.ModelType.TRANSFORMER,
        )
