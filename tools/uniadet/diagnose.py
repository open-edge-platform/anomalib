# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bounded FP32 scoring ablations using unchanged UniADet features and trained heads.

These are diagnostic alternatives, not replacement paper-reproduction results.
No variant is selected by optimizing target labels. The published executable
settings are pinned to gaobb/UniADet commit 2b81369d418ed08fd882452597ad5ebe7df3cb1d.
"""

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize

from anomalib.data import ImageBatch, InferenceBatch
from anomalib.data.datasets.image.mvtecad import CATEGORIES as MVTEC_CATEGORIES
from anomalib.data.datasets.image.mvtecad import MVTecADDataset
from anomalib.data.datasets.image.visa import CATEGORIES as VISA_CATEGORIES
from anomalib.data.datasets.image.visa import VisaDataset
from anomalib.data.utils import read_image
from anomalib.models import UniADet
from anomalib.models.image.uniadet.backbones import HierarchicalFeatures
from anomalib.models.image.uniadet.torch_model import UniADetModel

logger = logging.getLogger(__name__)


def scoring_variants(
    model: UniADetModel,
    features: HierarchicalFeatures,
) -> dict[str, InferenceBatch]:
    """Isolate score processing differences without changing the backbone or head.

    Args:
        model (UniADetModel): Trained detector with optional category reference memory.
        features (HierarchicalFeatures): Frozen query features from the detector.

    Returns:
        dict[str, InferenceBatch]: Baseline, one-variable probes, and the published
            executable combination. Maps have the encoder's input resolution.
    """
    logits = model.head(features, model.backbone.grid_size)
    image = torch.stack([value.softmax(-1)[:, 1] for value in logits.image]).mean(0)
    patches = torch.stack([value.softmax(1)[:, 1:2] for value in logits.pixel]).mean(0)
    size = (model.backbone.image_size,) * 2
    has_references = model.memory_bank.shape[1] > 0
    reference = model._reference_distances(features) if has_references else torch.zeros_like(patches)  # noqa: SLF001
    results = {}
    settings = {
        "baseline": (True, False, 0.5, 0.5, False, False),
        "resize_alignment_only": (False, False, 0.5, 0.5, False, False),
        "peak_resolution_only": (True, False, 0.5, 0.5, True, False),
        "recalibration_only": (True, True, 0.5, 0.5, False, False),
        "fusion_weights_only": (True, False, 0.8, 0.5 if has_references else 0.1, False, False),
        "reference_smoothing_only": (True, False, 0.5, 0.5, False, True),
        "published_executable": (False, True, 0.8, 0.5 if has_references else 0.1, True, True),
    }
    for name, (
        alignment,
        recalibrate,
        reference_weight,
        image_weight,
        resize_peak,
        smooth_reference,
    ) in settings.items():
        zero_map = functional.interpolate(patches, size=size, mode="bilinear", align_corners=alignment)
        reference_map = functional.interpolate(reference, size=size, mode="bilinear", align_corners=alignment)
        if recalibrate:
            zero_map = torch.sigmoid(15 * (zero_map - 0.4))
        if has_references:
            if smooth_reference:
                reference_map = model.blur(reference_map)
            if recalibrate:
                reference_map = torch.sigmoid(15 * (reference_map - 0.3))
            zero_map = (1 - reference_weight) * zero_map + reference_weight * reference_map
        anomaly_map = model.blur(zero_map)
        score_map = (
            functional.interpolate(anomaly_map, size=(256, 256), mode="bilinear", align_corners=False)
            if resize_peak
            else anomaly_map
        )
        score = (1 - image_weight) * image + image_weight * score_map.flatten(1).amax(1)
        results[name] = InferenceBatch(pred_score=score, anomaly_map=anomaly_map)
    results["image_head_only"] = InferenceBatch(pred_score=image, anomaly_map=results["baseline"].anomaly_map)
    return results


def main() -> None:
    """Compare predeclared inference variants on fixed cross-dataset categories and references."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mvtec", "visa"), default="mvtec")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--reference-sampling",
        choices=("without-replacement", "with-replacement"),
        default="without-replacement",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    manifest = json.loads((args.weights.parent / "training.json").read_text())
    if manifest["source_dataset"] == args.dataset:
        msg = "Auxiliary and target datasets must differ."
        raise ValueError(msg)
    allowed_categories = MVTEC_CATEGORIES if args.dataset == "mvtec" else VISA_CATEGORIES
    args.categories = args.categories or list(allowed_categories)
    if set(args.categories) - set(allowed_categories):
        msg = f"Categories must belong to {tuple(allowed_categories)}."
        raise ValueError(msg)
    dataset_class = MVTecADDataset if args.dataset == "mvtec" else VisaDataset
    if min(args.shots) < 0:
        msg = "Shot counts must be nonnegative."
        raise ValueError(msg)
    detector = (
        UniADet(
            backbone=manifest["backbone"],
            weights=args.weights,
            evaluator=False,
            visualizer=False,
            post_processor=False,
        )
        .to(args.device)
        .eval()
    )
    rows = []
    image_rows: list[dict[str, str | int | float]] = []
    references = {}
    largest_baseline_error = 0.0
    with torch.inference_mode():
        for category in args.categories:
            training = dataset_class(root=args.root, category=category, split="train")
            paths = sorted(training.samples.loc[training.samples.label_index == 0, "image_path"].tolist())
            banks = {}
            for shots in args.shots:
                detector.model.clear_reference_images()
                if shots:
                    generator = torch.Generator().manual_seed(args.seed)
                    selected = (
                        torch.randint(len(paths), (shots,), generator=generator).tolist()
                        if args.reference_sampling == "with-replacement"
                        else torch.randperm(len(paths), generator=generator)[:shots].tolist()
                    )
                    if len(selected) != shots:
                        msg = f"{category} has too few normal training images."
                        raise ValueError(msg)
                    references[f"{category}/{shots}"] = [
                        str(Path(paths[index]).relative_to(args.root.resolve())) for index in selected
                    ]
                    images = torch.cat([
                        detector.pre_processor(
                            read_image(paths[index], as_tensor=True).to(args.device).unsqueeze(0),
                        )
                        for index in selected
                    ])
                    detector.model.set_reference_images(images)
                banks[shots] = detector.model.memory_bank
            data = dataset_class(
                root=args.root,
                category=category,
                split="test",
                augmentations=Resize((detector.image_size,) * 2, antialias=False),
            )
            loader = DataLoader(data, batch_size=args.batch_size, num_workers=2, collate_fn=ImageBatch.collate)
            predictions: dict[tuple[int, str], list[torch.Tensor]] = {}
            labels = []
            category_paths: list[str] = []
            for batch in loader:
                features = detector.model.backbone(detector.pre_processor(batch.image.to(args.device)))
                labels.append(batch.gt_label)
                category_paths.extend(str(Path(path).relative_to(args.root.resolve())) for path in batch.image_path)
                for shots, bank in banks.items():
                    detector.model.memory_bank = bank
                    variants = scoring_variants(detector.model, features)
                    original = detector.model.predict_from_features(features)
                    error = (original.pred_score - variants["baseline"].pred_score).abs().max()
                    map_error = (original.anomaly_map - variants["baseline"].anomaly_map).abs().max()
                    largest_baseline_error = max(largest_baseline_error, float(error), float(map_error))
                    for name, prediction in variants.items():
                        predictions.setdefault((shots, name), []).append(prediction.pred_score.cpu())
            target = torch.cat(labels).numpy()
            for (shots, name), values in predictions.items():
                scores = torch.cat(values).numpy()
                rows.append({
                    "category": category,
                    "shots": shots,
                    "variant": name,
                    "image_auroc": roc_auc_score(target, scores),
                    "image_ap": average_precision_score(target, scores),
                })
                image_rows.extend(
                    {
                        "category": category,
                        "shots": shots,
                        "variant": name,
                        "path": path,
                        "label": int(label),
                        "score": float(score),
                    }
                    for path, label, score in zip(category_paths, target, scores, strict=True)
                )
            logger.info("Completed %s", category)
    if largest_baseline_error > 1e-5:
        msg = f"The diagnostic baseline differs from the detector by {largest_baseline_error}."
        raise RuntimeError(msg)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in (("categories.csv", rows), ("images.csv", image_rows)):
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    summary = {
        "diagnostic_only": True,
        "backbone": manifest["backbone"],
        "target_dataset": args.dataset,
        "head_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_revision": "2b81369d418ed08fd882452597ad5ebe7df3cb1d",
        "categories": args.categories,
        "shots": args.shots,
        "reference_seed": args.seed,
        "reference_sampling": args.reference_sampling,
        "references": references,
        "baseline_max_absolute_error": largest_baseline_error,
        "macro_results": [
            {
                "shots": shots,
                "variant": variant,
                **{
                    metric: sum(row[metric] for row in rows if row["shots"] == shots and row["variant"] == variant)
                    / len(args.categories)
                    for metric in ("image_auroc", "image_ap")
                },
            }
            for shots in args.shots
            for variant in sorted({row["variant"] for row in rows})
        ],
        "complete": True,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Results: %s", summary["macro_results"])


if __name__ == "__main__":
    main()
