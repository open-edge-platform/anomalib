# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Aggregate complete category sweeps and compare measured percentages with author-reported results."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from anomalib.data.datasets.image.mvtecad import CATEGORIES as MVTEC_CATEGORIES
from anomalib.data.datasets.image.visa import CATEGORIES as VISA_CATEGORIES

METRICS = ("image_AUROC", "image_AUPR", "pixel_AUROC", "pixel_AUPR")


def summarize_run(directory: Path) -> list[dict[str, str | int | float]]:
    """Compute macro-category means, then population standard deviations across reference seeds.

    Only one float64 ULP beyond a metric endpoint is clipped for reduction roundoff.
    Raw category measurements remain unchanged.

    Args:
        directory (Path): Completed benchmark directory containing category CSV and metadata.

    Returns:
        list[dict[str, str | int | float]]: One row per shot setting and metric, in percent.

    Raises:
        ValueError: If training or category/seed coverage is incomplete, duplicated, or inconsistent.
    """
    manifest = json.loads((directory / "evaluation.json").read_text())
    with (directory / "categories.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = set(MVTEC_CATEGORIES if manifest["target_dataset"] == "mvtec" else VISA_CATEGORIES)
    author = manifest.get("protocol") == "author-v1"
    valid_weights = (
        manifest["weights"]["origin"] == "author"
        and manifest["weights"]["source_dataset"] != manifest["target_dataset"]
        and manifest.get("paper_ap_definition") == "torchmetrics_auc_reorder"
        and manifest["shots"] == [0, 1, 2, 4]
        and manifest["seeds"] == [10, 20, 30]
        if author
        else manifest["training"]["complete_recipe"] and manifest["training"].get("recipe_version") == 2
    )
    if not manifest["complete"] or not valid_weights or set(manifest["categories"]) != expected:
        msg = f"{directory} is not a complete full-training, all-category benchmark."
        raise ValueError(msg)
    keys = [(row["category"], int(row["shots"]), int(row["seed"])) for row in rows]
    required = {
        (category, shots, seed)
        for category in expected
        for shots in manifest["shots"]
        for seed in (manifest["seeds"] if shots else manifest["seeds"][:1])
    }
    if len(keys) != len(set(keys)) or set(keys) != required:
        msg = f"{directory} has missing, duplicated, or unexpected category/shot/seed results."
        raise ValueError(msg)
    minimum, maximum = math.nextafter(0.0, -math.inf), math.nextafter(1.0, math.inf)
    if any(
        not math.isfinite(float(row[metric])) or not minimum <= float(row[metric]) <= maximum
        for row in rows
        for metric in METRICS
    ):
        msg = f"{directory} contains invalid or nonfinite metrics."
        raise ValueError(msg)
    summary = []
    for shots in manifest["shots"]:
        seeds = manifest["seeds"] if shots else manifest["seeds"][:1]
        for metric in METRICS:
            seed_means = [
                100
                * statistics.mean(
                    min(1.0, max(0.0, float(row[metric])))
                    for row in rows
                    if int(row["shots"]) == shots and int(row["seed"]) == seed
                )
                for seed in seeds
            ]
            summary.append({
                "backbone": manifest["weights" if author else "training"]["backbone"],
                "weight_origin": "author" if author else "independently-trained",
                "encoder_precision": manifest.get("encoder_precision", "fp32"),
                "aupr_definition": manifest.get("paper_ap_definition", "stepwise_average_precision"),
                "target_dataset": manifest["target_dataset"],
                "shots": shots,
                "metric": metric,
                "measured_mean": statistics.mean(seed_means),
                "measured_std": statistics.pstdev(seed_means),
                "categories": len(expected),
                "seeds": len(seeds),
            })
    return summary


def main() -> None:
    """Write measured summary and signed percentage-point gaps without filling missing runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with Path(__file__).with_name("paper_results.csv").open(newline="") as stream:
        paper = {(row["backbone"], row["target_dataset"], int(row["shots"])): row for row in csv.DictReader(stream)}
    rows = []
    for run in args.runs:
        for row in summarize_run(run):
            reference = paper[str(row["backbone"]), str(row["target_dataset"]), int(row["shots"])]
            author_result = float(reference[str(row["metric"])])
            rows.append({
                **row,
                "paper_mean": author_result,
                "gap_percentage_points": float(row["measured_mean"]) - author_result,
                "paper_source": reference["source"],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
