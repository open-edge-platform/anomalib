# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Cross-dataset UniADet evaluation with shared frozen features across shot settings.

The generic benchmark pipeline does not implement auxiliary supervised training
followed by multiple target shot counts. This narrow helper uses the integrated
model and anomalib metrics, reusing encoder outputs without persisting dataset features.
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import torch
from torch.nn import functional
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize

from anomalib.data import ImageBatch
from anomalib.data.datasets.image.mvtecad import CATEGORIES as MVTEC_CATEGORIES
from anomalib.data.datasets.image.mvtecad import MVTecADDataset
from anomalib.data.datasets.image.visa import CATEGORIES as VISA_CATEGORIES
from anomalib.data.datasets.image.visa import VisaDataset
from anomalib.data.utils import read_image
from anomalib.metrics import AUPR, AUROC
from anomalib.models import UniADet

logger = logging.getLogger(__name__)


def main() -> None:
    """Evaluate every target category, shot count, and seed without target test-set adaptation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mvtec", "visa"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 20, 30])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    manifest = json.loads((args.weights.parent / "training.json").read_text())
    if manifest.get("recipe_version") != 2:
        msg = "Retrain with corrected recipe version 2 before running a paper-protocol benchmark."
        raise ValueError(msg)
    if manifest["source_dataset"] == args.dataset:
        msg = "Auxiliary and target datasets must differ; same-dataset evaluation would leak test labels."
        raise ValueError(msg)
    model = (
        UniADet(
            backbone=manifest["backbone"],
            weights=args.weights,
            evaluator=False,
            visualizer=False,
            post_processor=False,
        )
        .to(f"cuda:{args.device}")
        .eval()
    )
    allowed_categories = MVTEC_CATEGORIES if args.dataset == "mvtec" else VISA_CATEGORIES
    categories = args.categories or allowed_categories
    if set(categories) - set(allowed_categories):
        msg = f"Categories must belong to {tuple(allowed_categories)}."
        raise ValueError(msg)
    if not args.shots or min(args.shots) < 0 or not args.seeds:
        msg = "Evaluation needs nonnegative shot counts and at least one seed."
        raise ValueError(msg)
    dataset_class = MVTecADDataset if args.dataset == "mvtec" else VisaDataset
    settings = [(shots, seed) for shots in args.shots for seed in (args.seeds if shots else args.seeds[:1])]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    references = {}
    start = time.perf_counter()
    with torch.inference_mode():
        for category in categories:
            train_data = dataset_class(root=args.root, category=category, split="train")
            paths = sorted(train_data.samples.loc[train_data.samples.label_index == 0, "image_path"].tolist())
            banks = {}
            for shots, seed in settings:
                model.model.clear_reference_images()
                if shots:
                    if len(paths) < shots:
                        msg = f"{category} has fewer than {shots} normal training images."
                        raise ValueError(msg)
                    generator = torch.Generator().manual_seed(seed)
                    indices = torch.randperm(len(paths), generator=generator)[:shots].tolist()
                    selected = [paths[index] for index in indices]
                    references[f"{category}/{shots}/{seed}"] = selected
                    images = torch.cat([
                        model.pre_processor(read_image(path, as_tensor=True).to(model.device).unsqueeze(0))
                        for path in selected
                    ])
                    model.model.set_reference_images(images)
                banks[shots, seed] = model.model.memory_bank
            metrics = {
                setting: [
                    AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
                    AUPR(fields=["pred_score", "gt_label"], prefix="image_"),
                    AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
                    AUPR(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
                ]
                for setting in settings
            }
            test_data = dataset_class(
                root=args.root,
                category=category,
                split="test",
                augmentations=Resize((model.image_size, model.image_size), antialias=False),
            )
            loader = DataLoader(
                test_data,
                batch_size=args.batch_size,
                num_workers=2,
                collate_fn=ImageBatch.collate,
            )
            for batch in loader:
                images = batch.image.to(model.device)
                features = model.model.backbone(model.pre_processor(images))
                masks = (
                    functional.interpolate(
                        batch.gt_mask.float().unsqueeze(1),
                        size=(256, 256),
                        mode="nearest",
                    )
                    .squeeze(1)
                    .bool()
                )
                for setting in settings:
                    model.model.memory_bank = banks[setting]
                    prediction = model.model.predict_from_features(features)
                    maps = functional.interpolate(
                        prediction.anomaly_map,
                        size=(256, 256),
                        mode="bilinear",
                        align_corners=True,
                    ).cpu()
                    result = ImageBatch(
                        image=batch.image,
                        gt_label=batch.gt_label,
                        gt_mask=masks,
                        pred_score=prediction.pred_score.cpu(),
                        anomaly_map=maps,
                    )
                    for metric in metrics[setting]:
                        metric.update(result)
            for (shots, seed), category_metrics in metrics.items():
                row = {
                    "backbone": manifest["backbone"],
                    "source_dataset": manifest["source_dataset"],
                    "target_dataset": args.dataset,
                    "category": category,
                    "shots": shots,
                    "seed": seed,
                    **{metric.name: float(metric.compute()) for metric in category_metrics},
                }
                rows.append(row)
                logger.info("%s", row)
            with (args.output / "categories.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    summary = {
        "implementation": "independent-paper",
        "training": manifest,
        "target_dataset": args.dataset,
        "categories": list(categories),
        "shots": args.shots,
        "seeds": args.seeds,
        "duration_seconds": time.perf_counter() - start,
        "pixel_resolution": 256,
        "few_shot_weight": model.model.few_shot_weight,
        "image_score_weight": model.model.image_score_weight,
        "complete": True,
        "references": references,
    }
    (args.output / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
