# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Clef-Flash decisions and image views used by VerdictAD.

Adapted from Shangquan Sun et al.'s VerdictAD-Jev-AD (Apache-2.0).
The optional decision model is loaded only on first inference.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from types import ModuleType

    from torch import nn
    from transformers.processing_utils import ProcessorMixin

Q_FREE = "Does the object in the image have a defect, damage, or contamination?"
Q_REF = (
    "Image 1 shows an object to inspect. Image 2 is a 2x2 grid of four normal reference samples "
    "of the same kind of object. "
    "Compared with the normal references, does the object in image 1 have a defect, damage, contamination, "
    "or a wrong arrangement?"
)
Q_CROP = (
    "Image 1 is a close-up crop of an object to inspect. Image 2 is a 2x2 grid of crops at the same location "
    "from four normal reference samples of the same kind of object. Compared with the normal references, "
    "does the crop in image 1 contain a defect, damage, contamination, or a wrong arrangement?"
)
Q_BOX = (
    "Image 1 is an object to inspect; a red box marks one area. Image 2 is a 2x2 grid of four normal reference samples "
    "of the same kind of object. Compared with the normal references, does the area inside the red box of image 1 "
    "show a defect, damage, contamination, a missing or extra item, or a wrong arrangement?"
)

# Integer fractions avoid windows extending beyond 1 through floating-point addition.
BOXES = [(x / 5, y / 5, (x + 2) / 5, (y + 2) / 5) for y in range(4) for x in range(4)]


def _resize(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.BICUBIC)


def _mosaic(images: list[Image.Image], size: int) -> Image.Image:
    """Repeat one to four references cyclically in a 2x2 grid."""
    result = Image.new("RGB", (size, size))
    half = size // 2
    for index in range(4):
        result.paste(_resize(images[index % len(images)], half), ((index % 2) * half, (index // 2) * half))
    return result


def _crop(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    return image.crop((int(box[0] * width), int(box[1] * height), int(box[2] * width), int(box[3] * height)))


def decision_records(
    image: Image.Image,
    references: list[Image.Image],
    *,
    include_global: bool = True,
    include_local: bool = True,
) -> list[tuple[str, list[Image.Image], str]]:
    """Build queries for the enabled refinement stages.

    Args:
        image (Image.Image): RGB query image.
        references (list[Image.Image]): One to four normal RGB reference images.
        include_global (bool): Whether to build the global queries.
        include_local (bool): Whether to build crop and boxed-view queries.

    Returns:
        list[tuple[str, list[Image.Image], str]]: Task, images and binary question per decision.

    Raises:
        ValueError: If the number of references is outside one to four.
    """
    if not 1 <= len(references) <= 4:
        msg = "VerdictAD needs one to four normal reference images."
        raise ValueError(msg)
    task = "Inspect an industrial product against normal references."
    records = []
    if include_global:
        query = _resize(image, 896)
        records.extend([
            ("Inspect this industrial product photo.", [query], Q_FREE),
            (task, [query, _mosaic(references, 896)], Q_REF),
        ])
    if include_local:
        records.extend(
            (task, [_resize(_crop(image, box), 448), _mosaic([_crop(r, box) for r in references], 448)], Q_CROP)
            for box in BOXES
        )
        reference_grid = _mosaic(references, 448)
        for box in BOXES:
            query = _resize(image, 448)
            coordinates = (int(box[0] * 448), int(box[1] * 448), int(box[2] * 448) - 1, int(box[3] * 448) - 1)
            ImageDraw.Draw(query).rectangle(coordinates, outline=(255, 0, 0), width=4)
            records.append((task, [query, reference_grid], Q_BOX))
    return records


class ClefFlash:
    """Lazy adapter for a trusted local Clef-Flash release.

    This executes the release's ``joint_schema_model.py``. Obtain and review the
    release separately; Anomalib neither downloads nor executes remote code on import.
    Weights remain external to Anomalib checkpoints.

    Args:
        model_path (str | None): Local release directory, including code and weights.
        batch_size (int): Number of decision records per forward pass.
    """

    def __init__(self, model_path: str | None = None, batch_size: int = 4) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            msg = "Decision batch_size must be positive."
            raise ValueError(msg)
        self.model_path = model_path
        self.batch_size = batch_size
        self._release: ModuleType | None = None
        self._model: nn.Module | None = None
        self._processor: ProcessorMixin | None = None

    def _load(self, device: torch.device) -> None:
        if self._model is not None:
            self._model.to(device).eval()
            return
        if self.model_path is None:
            msg = "Set decision_model_path to a trusted local Cloudflare/clef-flash release directory."
            raise ValueError(msg)
        path = Path(self.model_path).expanduser().resolve()
        source = path / "joint_schema_model.py"
        if not source.is_file():
            msg = f"Clef-Flash release code not found: {source}"
            raise FileNotFoundError(msg)
        name = "_anomalib_clef_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
        spec = importlib.util.spec_from_file_location(name, source)
        if spec is None or spec.loader is None:
            msg = f"Cannot load Clef-Flash release: {source}"
            raise ImportError(msg)
        release = importlib.util.module_from_spec(spec)
        sys.modules[name] = release  # Dataclasses in the release resolve annotations through sys.modules.
        try:
            spec.loader.exec_module(release)
            model, processor = release.load_release_model(
                path,
                device=device,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            )
        except Exception:
            sys.modules.pop(name, None)
            raise
        self._release, self._model, self._processor = release, model.eval(), processor

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        references: list[Image.Image],
        device: torch.device,
        *,
        include_global: bool = True,
        include_local: bool = True,
    ) -> torch.Tensor:
        """Return true-minus-false decision logits in global/crop/box order.

        Args:
            image (Image.Image): Query image.
            references (list[Image.Image]): Normal reference images.
            device (torch.device): Device on which to run the decision model.
            include_global (bool): Whether to run the two global decisions.
            include_local (bool): Whether to run the 32 local decisions.

        Returns:
            torch.Tensor: One float32 logit per requested decision on ``device``.
        """
        records = decision_records(image, references, include_global=include_global, include_local=include_local)
        if not records:
            return torch.empty(0, dtype=torch.float32, device=device)
        self._load(device)
        if self._release is None or self._model is None or self._processor is None:
            msg = "Clef-Flash did not initialize its model and processor."
            raise RuntimeError(msg)
        result = []
        for start in range(0, len(records), self.batch_size):
            encoded = [
                self._release.encode_record(
                    self._processor.tokenizer,
                    {
                        "state": {"task": task},
                        "images": images,
                        "questions": {"defect": {"type": "noul", "instructions": question}},
                    },
                    processor=self._processor,
                )
                for task, images, question in records[start : start + self.batch_size]
            ]
            batch = self._release.collate_records(encoded, self._processor.tokenizer.pad_token_id, device)
            outputs = self._model(batch)
            for record, output in zip(encoded, outputs, strict=True):
                options = list(record.questions[0].option_ids)
                logits = output[0].float()
                result.append(logits[options.index("true")] - logits[options.index("false")])
        return torch.stack(result).to(device)
