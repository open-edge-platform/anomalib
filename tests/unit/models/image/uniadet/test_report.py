# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests that benchmark summaries cannot silently publish partial or duplicated measurements."""

import csv
import json
import math
from pathlib import Path

import pytest
from tools.uniadet.report import METRICS, summarize_run

from anomalib.data.datasets.image.mvtecad import CATEGORIES


@pytest.fixture
def benchmark_run(tmp_path: Path) -> Path:
    """Create synthetic all-category records, only for testing report aggregation."""
    manifest = {
        "complete": True,
        "training": {"complete_recipe": True, "backbone": "dinov2", "recipe_version": 2},
        "target_dataset": "mvtec",
        "categories": list(CATEGORIES),
        "shots": [1],
        "seeds": [10, 20],
    }
    (tmp_path / "evaluation.json").write_text(json.dumps(manifest))
    rows = [
        {"category": category, "shots": 1, "seed": seed, **dict.fromkeys(METRICS, value)}
        for category in CATEGORIES
        for seed, value in [(10, 0.6), (20, 0.8)]
    ]
    with (tmp_path / "categories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def test_macro_summary_and_seed_variation(benchmark_run: Path) -> None:
    """Reference-seed variation is computed after category averaging, in percent."""
    for row in summarize_run(benchmark_run):
        assert row["measured_mean"] == pytest.approx(70)
        assert row["measured_std"] == pytest.approx(10)


@pytest.mark.parametrize(
    "value",
    [math.nextafter(1.0, math.inf), 1.0 + 2 * math.ulp(1.0), 1.000001, -0.000001],
)
def test_metric_roundoff_is_bounded_to_one_ulp(benchmark_run: Path, value: float) -> None:
    """Perfect sklearn AP can exceed one by a float64 ULP; genuine out-of-range values remain invalid."""
    path = benchmark_run / "categories.csv"
    path.write_text(path.read_text().replace("0.8", str(value)))
    if value == math.nextafter(1.0, math.inf):
        for row in summarize_run(benchmark_run):
            assert row["measured_mean"] == pytest.approx(80)
            assert row["measured_std"] == pytest.approx(20)
    else:
        with pytest.raises(ValueError, match="invalid or nonfinite"):
            summarize_run(benchmark_run)


@pytest.mark.parametrize("version", [None, 1])
def test_reject_outdated_training_recipe(benchmark_run: Path, version: int | None) -> None:
    """Initial-recipe measurements cannot masquerade as corrected author-parameter runs."""
    path = benchmark_run / "evaluation.json"
    manifest = json.loads(path.read_text())
    manifest["training"]["recipe_version"] = version
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete full-training"):
        summarize_run(benchmark_run)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "nonfinite"])
def test_reject_incomplete_or_invalid_measurements(benchmark_run: Path, corruption: str) -> None:
    """Missing/duplicated categories or NaN scores are not silently averaged into headline tables."""
    path = benchmark_run / "categories.csv"
    lines = path.read_text().splitlines()
    if corruption == "missing":
        lines.pop()
    elif corruption == "duplicate":
        lines.append(lines[-1])
    else:
        lines[-1] = lines[-1].replace("0.8", "nan")
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=r"missing|duplicated|nonfinite"):
        summarize_run(benchmark_run)


def test_author_evaluation_is_not_reported_as_independent_training(tmp_path: Path) -> None:
    """Complete author-head inference is accepted without inventing a locally reproduced training run."""
    manifest = {
        "complete": True,
        "protocol": "author-v1",
        "paper_ap_definition": "torchmetrics_auc_reorder",
        "weights": {"origin": "author", "backbone": "dinov2", "source_dataset": "visa"},
        "target_dataset": "mvtec",
        "categories": list(CATEGORIES),
        "shots": [0, 1, 2, 4],
        "seeds": [10, 20, 30],
    }
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(manifest))
    rows = [
        {"category": category, "shots": shots, "seed": seed, **dict.fromkeys(METRICS, value)}
        for category in CATEGORIES
        for shots in manifest["shots"]
        for seed, value in ([(10, 0.6)] if shots == 0 else [(10, 0.6), (20, 0.8), (30, 1.0)])
    ]
    with (tmp_path / "categories.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_run(tmp_path)
    assert len(summary) == 16
    assert all(row["weight_origin"] == "author" for row in summary)
    for row in summary:
        assert row["measured_mean"] == pytest.approx(60 if row["shots"] == 0 else 80)
    manifest["paper_ap_definition"] = "stepwise_average_precision"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete full-training"):
        summarize_run(tmp_path)
    manifest["paper_ap_definition"] = "torchmetrics_auc_reorder"
    manifest["weights"]["source_dataset"] = "mvtec"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete full-training"):
        summarize_run(tmp_path)
