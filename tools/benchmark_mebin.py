# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark MEBinPostProcessor vs default PostProcessor.

Trains PatchCore, PaDiM, and AnomalyDINO on a selection of MVTec AD categories and compares
the pixel-level and image-level metrics between the default PostProcessor
(F1-adaptive threshold) and the MEBinPostProcessor.

Usage:
    python tools/benchmark_mebin.py [--categories bottle carpet ...] [--root ./datasets/MVTecAD]

Results are printed as a table and saved to    results/mebin_benchmark.csv   .
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from lightning import seed_everything

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import AnomalyDINO, Padim, Patchcore
from anomalib.post_processing import MEBinPostProcessor, PostProcessor

logger = logging.getLogger(__name__)

# A representative subset of MVTec categories covering textures (carpet, grid,
# leather) and objects (bottle, cable, hazelnut) to keep runtime reasonable.
DEFAULT_CATEGORIES = ["bottle", "cable", "carpet", "hazelnut", "leather", "grid"]


def _run_experiment(
    model_cls: type,
    model_name: str,
    category: str,
    post_processor: PostProcessor,
    pp_name: str,
    root: str,
) -> dict:
    """Run a single train+test experiment and return metrics."""
    datamodule = MVTecAD(
        root=root,
        category=category,
    )
    model = model_cls(post_processor=post_processor)

    engine = Engine(
        default_root_dir=f"results/mebin_benchmark/{model_name}_{pp_name}_{category}",
        barebones=True,
    )

    engine.fit(model=model, datamodule=datamodule)
    test_results = engine.test(model=model, datamodule=datamodule)

    metrics = test_results[0] if test_results else {}
    return {
        "model": model_name,
        "category": category,
        "post_processor": pp_name,
        "image_AUROC": metrics.get("image_AUROC", float("nan")),
        "image_F1Score": metrics.get("image_F1Score", float("nan")),
        "pixel_AUROC": metrics.get("pixel_AUROC", float("nan")),
        "pixel_F1Score": metrics.get("pixel_F1Score", float("nan")),
    }


def main() -> None:
    """Run the benchmark."""
    seed_everything(117)

    parser = argparse.ArgumentParser(description="Benchmark MEBin vs default PostProcessor")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="MVTec AD categories to benchmark",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./datasets/MVTecAD",
        help="Root path to MVTec AD dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/mebin_benchmark.csv",
        help="Path to save CSV results",
    )
    args = parser.parse_args()

    model_configs: list[tuple[type, str]] = [
        (Patchcore, "PatchCore"),
        (Padim, "PaDiM"),
        (AnomalyDINO, "AnomalyDINO"),
    ]

    post_processor_configs: list[tuple[PostProcessor | MEBinPostProcessor, str]] = [
        (PostProcessor(), "Default"),
        (MEBinPostProcessor(sample_rate=4, min_interval_len=4, erode=True, kernel_size=6), "MEBin"),
    ]

    results: list[dict] = []

    for model_cls, model_name in model_configs:
        for category in args.categories:
            for post_processor, processor_name in post_processor_configs:
                logger.info("Running: %s / %s / %s", model_name, category, processor_name)
                try:
                    row = _run_experiment(model_cls, model_name, category, post_processor, processor_name, args.root)
                    results.append(row)
                    logger.info(
                        "  -> image_AUROC=%.4f  pixel_AUROC=%.4f  pixel_F1=%.4f",
                        row["image_AUROC"],
                        row["pixel_AUROC"],
                        row["pixel_F1Score"],
                    )
                except Exception:
                    logger.exception("  -> FAILED: %s / %s / %s", model_name, category, processor_name)

    header = ["model", "category", "post_processor", "image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"]
    col_widths = [max(len(h), max((len(str(r.get(h, ""))) for r in results), default=0)) + 2 for h in header]
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)

    print("\n" + "=" * sum(col_widths))
    print(fmt.format(*header))
    print("-" * sum(col_widths))
    for row in results:
        print(fmt.format(*[f"{row[h]:.4f}" if isinstance(row[h], float) else row[h] for h in header]))
    print("=" * sum(col_widths))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
