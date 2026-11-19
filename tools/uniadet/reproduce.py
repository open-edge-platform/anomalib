# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Evaluate authorized author heads with the pinned industrial evaluation protocol.

Uses original VisA 1cls.csv ordering, normal MVTec training images, with-replacement
reference seeds 10/20/30, OpenCV uint8 bilinear resizing and 256-square pixel metrics.
This reproduces evaluation with author weights, not auxiliary training from scratch.
"""

import argparse
import csv
import hashlib
import json
import logging
import platform
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import timm
import torch
import torchmetrics
from PIL import Image
from timm.layers import LayerNormFp32
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.classification import binary_auroc, binary_precision_recall_curve
from torchmetrics.utilities.compute import auc
from torchvision.transforms.functional import gaussian_blur

from anomalib.data import ImageBatch, ImageItem
from anomalib.data.datasets.image.mvtecad import CATEGORIES as MVTEC_CATEGORIES
from anomalib.data.datasets.image.visa import CATEGORIES as VISA_CATEGORIES
from anomalib.data.utils.path import resolve_path_under_root
from anomalib.models import UniADet

logger = logging.getLogger(__name__)


class _AuthorBlur(torch.nn.Module):
    """Keep torchvision's input-dtype Gaussian kernel for the native mixed-precision probe."""

    def __init__(self) -> None:
        super().__init__()
        self.kernel_size = [33, 33]
        self.sigma = [4.0, 4.0]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return gaussian_blur(image, kernel_size=self.kernel_size, sigma=self.sigma)


class Sample(NamedTuple):
    """Confined original image/mask paths and the binary image label."""

    image: Path
    mask: Path | None
    label: int


def category_samples(root: Path, dataset: str, category: str, split: str) -> list[Sample]:
    """Read the author's original public split and preserve its normal-reference pool order.

    Args:
        root (Path): Original dataset directory, not prepared ``visa_pytorch``.
        dataset (str): ``"mvtec"`` or ``"visa"``.
        category (str): Allowlisted object category.
        split (str): ``"train"`` or ``"test"``.

    Returns:
        list[Sample]: Confined image and mask paths in the source metadata order.

    Raises:
        ValueError: If the protocol, split, label, category or a metadata path is invalid.
    """
    allowed = MVTEC_CATEGORIES if dataset == "mvtec" else VISA_CATEGORIES
    if dataset not in {"mvtec", "visa"} or category not in allowed or split not in {"train", "test"}:
        msg = "Invalid dataset, category or split."
        raise ValueError(msg)
    root = root.resolve()
    samples = []
    if dataset == "visa":
        split_path = resolve_path_under_root(root, "split_csv/1cls.csv")
        with split_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["object"] != category or row["split"] != split:
                    continue
                if row["label"] not in {"normal", "anomaly"}:
                    msg = "VisA labels must be normal or anomaly."
                    raise ValueError(msg)
                anomalous = row["label"] == "anomaly"
                samples.append(
                    Sample(
                        resolve_path_under_root(root, row["image"]),
                        resolve_path_under_root(root, row["mask"]) if anomalous else None,
                        int(anomalous),
                    ),
                )
    else:
        directory = resolve_path_under_root(root, f"{category}/{split}")
        for defect in sorted(directory.iterdir()):
            resolve_path_under_root(root, defect)
            for image in sorted(defect.glob("*.png")):
                anomalous = defect.name != "good"
                mask = (
                    resolve_path_under_root(root, f"{category}/ground_truth/{defect.name}/{image.stem}_mask.png")
                    if anomalous
                    else None
                )
                samples.append(Sample(resolve_path_under_root(root, image), mask, int(anomalous)))
    if not samples or (split == "train" and any(sample.label for sample in samples)):
        msg = "Expected a nonempty split and exclusively normal target training references."
        raise ValueError(msg)
    return samples


class EvaluationDataset(Dataset):
    """Decode original RGB files and resize image bytes/masks before tensor normalization.

    Args:
        samples (list[Sample]): Validated public split records.
        image_size (int): Square encoder input resolution.
    """

    def __init__(self, samples: list[Sample], image_size: int) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        """Return the number of images in the split."""
        return len(self.samples)

    def __getitem__(self, index: int) -> ImageItem:
        """Decode and resize an original image and its binary mask.

        Args:
            index (int): Index into the validated split.

        Returns:
            ImageItem: Unnormalized float RGB image, nearest-resized mask and image label.
        """
        sample = self.samples[index]
        with Image.open(sample.image) as original:
            image = np.array(original.convert("RGB"))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if sample.mask is not None:
            with Image.open(sample.mask) as original_mask:
                mask = (np.array(original_mask.convert("L")) > 0).astype(np.uint8)
        size = (self.image_size,) * 2
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        return ImageItem(
            image=torch.from_numpy(image).permute(2, 0, 1).float() / 255,
            gt_mask=torch.from_numpy(mask).bool(),
            gt_label=sample.label,
            image_path=str(sample.image),
        )


def reference_indices(count: int, shots: int, seed: int) -> list[int]:
    """Draw the published per-category CPU randint sequence without modifying global RNG state.

    Args:
        count (int): Number of normal training images in original metadata order.
        shots (int): Reference count; duplicate draws are retained.
        seed (int): Explicit reference seed.

    Returns:
        list[int]: With-replacement indices into the original normal pool.

    Raises:
        ValueError: If the pool is empty or the shot count is negative.
    """
    if count <= 0 or shots < 0:
        msg = "Reference sampling requires a nonempty pool and nonnegative shots."
        raise ValueError(msg)
    return torch.randint(count, (shots,), generator=torch.Generator().manual_seed(seed)).tolist()


def use_author_clip_precision(encoder: torch.nn.Module) -> None:
    """Use FP16 projections and FP32 LayerNorm without rounding its original parameters.

    Args:
        encoder (torch.nn.Module): Frozen CLIP vision encoder to configure in place.
    """
    normalization = [
        (name, module, {key: value.detach().clone() for key, value in module.state_dict().items()})
        for name, module in encoder.named_modules()
        if isinstance(module, torch.nn.LayerNorm)
    ]
    device = next(encoder.parameters()).device
    encoder.half()
    for name, module, state in normalization:
        replacement = LayerNormFp32(
            module.normalized_shape[0],
            eps=module.eps,
            affine=module.elementwise_affine,
            device=device,
        )
        replacement.load_state_dict(state)
        replacement.requires_grad_(requires_grad=False).eval()
        encoder.set_submodule(name, replacement)


def binary_metrics(predictions: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Keep the released evaluator's PR integration separate from standard average precision.

    The author's ``AP`` is trapezoidal PR area after sorting recall. This is not
    stepwise average precision, especially for tied scores. Neither convention
    changes anomalib's standard metrics.

    Args:
        predictions (torch.Tensor): Finite continuous scores, on the evaluation device.
        target (torch.Tensor): Corresponding binary labels, with the same shape.

    Returns:
        dict[str, float]: Exact AUROC, paper-specific AUPR, and standard average precision.
    """
    predictions, target = predictions.flatten(), target.flatten().int()
    precision, recall, _ = binary_precision_recall_curve(predictions, target)
    return {
        "AUROC": binary_auroc(predictions, target).item(),
        "AUPR": auc(recall, precision, reorder=True).item(),
        "average_precision": (-torch.sum((recall[1:] - recall[:-1]) * precision[:-1])).item(),
    }


def main() -> None:
    """Evaluate every category, shot and seed, recording input identity and weight provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mvtec", "visa"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-precision", choices=("fp32", "author"), default="fp32")
    parser.add_argument("--categories", nargs="+", help="Smoke subset; never a complete benchmark.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    torch.manual_seed(10)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    provenance = json.loads((args.weights.parent / "provenance.json").read_text())
    if (
        provenance.get("origin") != "author"
        or provenance.get("protocol") != "author-v1"
        or provenance["source_dataset"] == args.dataset
        or hashlib.sha256(args.weights.read_bytes()).hexdigest() != provenance["head_sha256"]
    ):
        msg = "Reproduction requires a verified author head trained on the other dataset."
        raise ValueError(msg)
    allowed = MVTEC_CATEGORIES if args.dataset == "mvtec" else VISA_CATEGORIES
    categories = args.categories or list(allowed)
    if set(categories) - set(allowed) or len(set(categories)) != len(categories):
        msg = "Categories must be unique members of the target dataset."
        raise ValueError(msg)
    model = (
        UniADet(
            backbone=provenance["backbone"],
            weights=args.weights,
            protocol="author",
            evaluator=False,
            visualizer=False,
            post_processor=False,
        )
        .to(f"cuda:{args.device}")
        .eval()
    )
    if args.encoder_precision == "author" and provenance["backbone"] == "clip":
        use_author_clip_precision(model.model.backbone.encoder)
        model.model.blur = _AuthorBlur()
    settings = [(0, 10), *((shots, seed) for shots in (1, 2, 4) for seed in (10, 20, 30))]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    references = {}
    splits = {}
    start = time.perf_counter()
    source_files = [
        Path(__file__),
        Path(__file__).with_name("author.py"),
        Path("src/anomalib/models/image/uniadet") / "backbones.py",
        Path("src/anomalib/models/image/uniadet") / "head.py",
        Path("src/anomalib/models/image/uniadet") / "torch_model.py",
        Path("src/anomalib/models/image/uniadet") / "lightning_model.py",
    ]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    with torch.inference_mode():
        for category in categories:
            training = category_samples(args.root, args.dataset, category, "train")
            testing = category_samples(args.root, args.dataset, category, "test")
            split_manifest = {
                split: [
                    {
                        "image": str(sample.image.relative_to(args.root.resolve())),
                        "mask": str(sample.mask.relative_to(args.root.resolve())) if sample.mask else None,
                        "label": sample.label,
                    }
                    for sample in samples
                ]
                for split, samples in (("train", training), ("test", testing))
            }
            splits[category] = {
                "train": len(training),
                "test": len(testing),
                "test_normal": sum(sample.label == 0 for sample in testing),
                "test_anomalous": sum(sample.label == 1 for sample in testing),
                "ordered_split_sha256": hashlib.sha256(json.dumps(split_manifest).encode()).hexdigest(),
            }
            banks = {}
            for shots, seed in settings:
                model.model.clear_reference_images()
                if shots:
                    selected = [training[index] for index in reference_indices(len(training), shots, seed)]
                    references[f"{category}/{shots}/{seed}"] = [
                        str(sample.image.relative_to(args.root.resolve())) for sample in selected
                    ]
                    images = torch.stack([item.image for item in EvaluationDataset(selected, model.image_size)])
                    model.model.set_reference_images(model.pre_processor(images.to(model.device)))
                banks[shots, seed] = model.model.memory_bank
            loader = DataLoader(
                EvaluationDataset(testing, model.image_size),
                batch_size=args.batch_size,
                num_workers=2,
                collate_fn=ImageBatch.collate,
            )
            scores: dict[tuple[int, int], list[torch.Tensor]] = {setting: [] for setting in settings}
            maps: dict[tuple[int, int], list[torch.Tensor]] = {setting: [] for setting in settings}
            labels, masks = [], []
            for batch in loader:
                features = model.model.backbone(model.pre_processor(batch.image.to(model.device)))
                labels.append(batch.gt_label)
                masks.append(functional.interpolate(batch.gt_mask[:, None].float(), (256, 256), mode="nearest").bool())
                for setting in settings:
                    model.model.memory_bank = banks[setting]
                    prediction = model.model.predict_from_features(features)
                    scores[setting].append(prediction.pred_score.cpu())
                    maps[setting].append(
                        functional.interpolate(
                            prediction.anomaly_map,
                            (256, 256),
                            mode="bilinear",
                            align_corners=False,
                        ).cpu(),
                    )
            target = torch.cat(labels).to(model.device)
            pixel_target = torch.cat(masks).to(model.device)
            for shots, seed in settings:
                image_scores = torch.cat(scores[shots, seed]).to(model.device)
                pixel_scores = torch.cat(maps[shots, seed]).to(model.device)
                if not torch.isfinite(image_scores).all() or not torch.isfinite(pixel_scores).all():
                    msg = "Nonfinite predictions invalidate the reproduction."
                    raise RuntimeError(msg)
                image_metrics = binary_metrics(image_scores, target)
                pixel_metrics = binary_metrics(pixel_scores, pixel_target)
                rows.append({
                    "backbone": provenance["backbone"],
                    "source_dataset": provenance["source_dataset"],
                    "target_dataset": args.dataset,
                    "category": category,
                    "shots": shots,
                    "seed": seed,
                    "image_AUROC": image_metrics["AUROC"],
                    "image_AUPR": image_metrics["AUPR"],
                    "pixel_AUROC": pixel_metrics["AUROC"],
                    "pixel_AUPR": pixel_metrics["AUPR"],
                    "image_average_precision": image_metrics["average_precision"],
                    "pixel_average_precision": pixel_metrics["average_precision"],
                })
                logger.info("%s", rows[-1])
            with (args.output / "categories.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Completed category %s (%d/%d)", category, categories.index(category) + 1, len(categories))
    manifest = {
        "protocol": "author-v1",
        "weights": provenance,
        "target_dataset": args.dataset,
        "categories": categories,
        "shots": [0, 1, 2, 4],
        "seeds": [10, 20, 30],
        "batch_size": args.batch_size,
        "pixel_resolution": 256,
        "reference_sampling": "CPU torch.randint with replacement, reseeded per category",
        "preprocessing": (
            "PIL RGB decode; OpenCV uint8 INTER_LINEAR image and INTER_NEAREST mask resize; backbone normalization"
        ),
        "encoder_precision": args.encoder_precision,
        "precision_details": (
            "CLIP encoder FP16 with FP32 LayerNorm and prototypes; DINO encoders FP32"
            if args.encoder_precision == "author"
            else "FP32 throughout; no autocast or TF32"
        ),
        "metric_backend": "Torchmetrics exact AUROC/PR curve; paper recall-sorted trapezoid and separate standard AP",
        "paper_ap_definition": "torchmetrics_auc_reorder",
        "metric_device": str(model.device),
        "aggregation": "category macro mean per seed, then seed mean and population standard deviation",
        "training_reproduced": False,
        "implementation_sha256": source_hashes,
        "references": references,
        "splits": splits,
        "visa_split_sha256": (
            hashlib.sha256((args.root / "split_csv/1cls.csv").read_bytes()).hexdigest()
            if args.dataset == "visa"
            else None
        ),
        "torch": torch.__version__,
        "timm": timm.__version__,
        "torchmetrics": torchmetrics.__version__,
        "opencv": cv2.__version__,
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name(args.device),
        "duration_seconds": time.perf_counter() - start,
        "complete": set(categories) == set(allowed),
    }
    (args.output / "evaluation.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
