# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test working of tiled ensemble pipeline components."""

import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from anomalib.data import ImageBatch, get_datamodule
from anomalib.pipelines.tiled_ensemble.components import (
    MergeJobGenerator,
    MetricsCalculationJobGenerator,
    NormalizationJobGenerator,
    SmoothingJobGenerator,
    StatisticsJobGenerator,
    ThresholdingJobGenerator,
)
from anomalib.pipelines.tiled_ensemble.components.metrics_calculation import MetricsCalculationJob
from anomalib.pipelines.tiled_ensemble.components.smoothing import SmoothingJob
from anomalib.pipelines.tiled_ensemble.components.utils import NormalizationStage
from anomalib.pipelines.tiled_ensemble.components.utils.helper_functions import setup_transforms
from anomalib.pipelines.tiled_ensemble.components.utils.prediction_data import EnsemblePredictions
from anomalib.pipelines.tiled_ensemble.components.utils.prediction_merging import PredictionMergingMechanism


class TestMerging:
    """Test merging mechanism and merging job."""

    @staticmethod
    def test_tile_merging(get_ensemble_config: dict, get_merging_mechanism: PredictionMergingMechanism) -> None:
        """Test tiled data merging."""
        config = get_ensemble_config
        merger = get_merging_mechanism

        # prepared original data
        datamodule = get_datamodule(config)
        datamodule.setup()
        # to ensure that ensemble data image size matches reference data
        setup_transforms(datamodule, config["tiling"]["image_size"])
        original_data = next(iter(datamodule.test_dataloader()))

        batch = merger.ensemble_predictions.get_batch_tiles(0)

        merged_image = merger.merge_tiles(batch, "image")
        assert merged_image.equal(original_data.image)

        merged_mask = merger.merge_tiles(batch, "gt_mask")
        assert merged_mask.equal(original_data.gt_mask)

    @staticmethod
    def test_label_and_score_merging(get_merging_mechanism: PredictionMergingMechanism) -> None:
        """Test label and score merging."""
        merger = get_merging_mechanism
        scores = torch.rand(4, 10)
        labels = scores > 0.5
        mock_img = torch.rand(3, 10, 10)

        mock_data = {}
        for i, idx in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            mock_data[idx] = ImageBatch(image=mock_img, pred_score=scores[i], pred_label=labels[i])

        merged = merger.merge_labels_and_scores(mock_data)

        assert merged["pred_score"].equal(scores.mean(dim=0))

        assert merged["pred_label"].equal(labels.any(dim=0))

    @staticmethod
    def test_all_merged(get_merging_mechanism: PredictionMergingMechanism) -> None:
        """Test that all keys are present in merged output."""
        merging_mechanism = get_merging_mechanism
        merged_direct = merging_mechanism.merge_tile_predictions(0)

        for key in [
            "image_path",
            "mask_path",
            "gt_label",
            "gt_mask",
            "image",
            "anomaly_map",
            "pred_mask",
            "pred_label",
            "pred_score",
        ]:
            assert hasattr(merged_direct, key)

    @staticmethod
    def test_merge_job(
        get_tile_predictions: EnsemblePredictions,
        get_ensemble_config: dict,
        get_merging_mechanism: PredictionMergingMechanism,
    ) -> None:
        """Test merging job execution."""
        config = get_ensemble_config
        predictions = copy.deepcopy(get_tile_predictions)
        merging_mechanism = get_merging_mechanism

        merging_job_generator = MergeJobGenerator(tiling_args=config["tiling"], data_args=config["data"])
        merging_job = next(merging_job_generator.generate_jobs(prev_stage_result=predictions))

        merged_direct = merging_mechanism.merge_tile_predictions(0)
        merged_with_job = merging_job.run()[0]

        # check that merging by job is same as with the mechanism directly
        for name in merged_direct.__dict__:
            value = getattr(merged_direct, name)
            job_value = getattr(merged_with_job, name)
            if isinstance(value, torch.Tensor):
                assert job_value.equal(value)
            else:
                assert job_value == value


class TestStatsCalculation:
    """Test post-processing statistics calculations."""

    @staticmethod
    def test_stats_run(project_path: Path) -> None:
        """Test execution of statistics calc. job."""
        mock_preds = [
            {
                "image": torch.rand(5, 3, 50, 50),
                "pred_score": torch.rand(4),
                "gt_label": torch.ones(4, dtype=torch.int32),
                "anomaly_map": torch.rand(4, 1, 50, 50),
                "gt_mask": torch.ones(4, 1, 50, 50, dtype=torch.int32),
            },
        ]
        data = [ImageBatch(**values) for values in mock_preds]

        stats_job_generator = StatisticsJobGenerator(project_path)
        stats_job = next(stats_job_generator.generate_jobs(None, data))

        results = stats_job.run()

        assert "minmax" in results
        assert "image_threshold" in results
        assert "pixel_threshold" in results

        # save as it's removed from results
        save_path = results["save_path"]
        stats_job.save(results)
        assert Path(save_path).exists()

    @staticmethod
    @pytest.mark.parametrize(
        ("key", "values"),
        [
            ("anomaly_map", [torch.rand(5, 1, 50, 50), torch.rand(5, 1, 50, 50)]),
            ("pred_score", [torch.rand(5), torch.rand(5)]),
        ],
    )
    def test_minmax(key: str, values: list) -> None:
        """Test minmax stats calculation."""
        # add given keys to test all possible sources of minmax
        data = [
            {
                "image": torch.rand(5, 3, 50, 50),
                "pred_score": torch.rand(5),
                "gt_label": torch.ones(5, dtype=torch.int32),
                key: values[0],
            },
            {
                "image": torch.rand(5, 3, 50, 50),
                "pred_score": torch.rand(5),
                "gt_label": torch.ones(5, dtype=torch.int32),
                key: values[1],
            },
        ]
        data = [ImageBatch(**values) for values in data]

        stats_job_generator = StatisticsJobGenerator(Path("mock"))
        stats_job = next(stats_job_generator.generate_jobs(None, data))
        results = stats_job.run()

        if isinstance(values[0], list):
            values[0] = torch.cat(values[0])
            values[1] = torch.cat(values[1])
        values = torch.stack(values)

        assert results["minmax"][key]["min"] == torch.min(values)
        assert results["minmax"][key]["max"] == torch.max(values)

    @staticmethod
    @pytest.mark.parametrize(
        ("gt_label", "preds", "target_threshold"),
        [
            (
                torch.tensor([0, 0, 0, 1, 1]).type(torch.int32),
                torch.tensor([2.3, 1.6, 2.6, 7.9, 3.3]),
                3.3,
            ),  # standard case
            (
                torch.tensor([1, 0, 0, 0]).type(torch.int32),
                torch.tensor([4, 3, 2, 1]),
                4,
            ),  # 100% recall for all thresholds
        ],
    )
    def test_threshold(gt_label: torch.Tensor, preds: torch.Tensor, target_threshold: float) -> None:
        """Test threshold calculation job."""
        data = [
            {
                "image": torch.rand(5, 3, 50, 50),
                "gt_label": gt_label,
                "gt_mask": torch.rand(5, 50, 50) > 0.5,
                "pred_score": preds,
                "anomaly_map": torch.rand(5, 50, 50),
            },
        ]
        data = [ImageBatch(**values) for values in data]

        stats_job_generator = StatisticsJobGenerator(Path("mock"))
        stats_job = next(stats_job_generator.generate_jobs(None, data))
        results = stats_job.run()

        assert round(results["image_threshold"], 5) == target_threshold
        # pixel threshold is not nan
        assert results["pixel_threshold"] == results["pixel_threshold"]


class TestMetrics:
    """Test ensemble metrics."""

    @pytest.fixture(scope="class")
    @staticmethod
    def get_ensemble_metrics_job(
        get_ensemble_config: dict,
        get_batch_predictions: list[dict],
    ) -> tuple[MetricsCalculationJob, str]:
        """Return Metrics calculation job and path to directory where metrics csv will be saved."""
        config = get_ensemble_config
        with TemporaryDirectory() as tmp_dir:
            metrics = MetricsCalculationJobGenerator(
                config["accelerator"],
                root_dir=Path(tmp_dir),
                model_args=config["TrainModels"]["model"],
            )

        mock_predictions = get_batch_predictions

        return next(metrics.generate_jobs(prev_stage_result=copy.deepcopy(mock_predictions))), tmp_dir

    @staticmethod
    def test_metrics_result(get_ensemble_metrics_job: tuple[MetricsCalculationJob, str]) -> None:
        """Test metrics result."""
        metrics_job, _ = get_ensemble_metrics_job

        result = metrics_job.run()

        assert "pixel_AUROC" in result
        assert "image_AUROC" in result

    @staticmethod
    def test_metrics_saving(get_ensemble_metrics_job: tuple[MetricsCalculationJob, str]) -> None:
        """Test metrics saving to csv."""
        metrics_job, tmp_dir = get_ensemble_metrics_job

        result = metrics_job.run()
        metrics_job.save(result)
        assert (Path(tmp_dir) / "metric_results.csv").exists()


class TestJoinSmoothing:
    """Test JoinSmoothing job responsible for smoothing area at tile seams."""

    @pytest.fixture(scope="class")
    @staticmethod
    def get_join_smoothing_job(get_ensemble_config: dict, get_batch_predictions: list[ImageBatch]) -> SmoothingJob:
        """Make and return SmoothingJob instance."""
        config = get_ensemble_config
        job_gen = SmoothingJobGenerator(
            accelerator=config["accelerator"],
            tiling_args=config["tiling"],
            data_args=config["data"],
        )
        # copy since smoothing changes data
        mock_predictions = copy.deepcopy(get_batch_predictions)
        return next(job_gen.generate_jobs(config["SeamSmoothing"], mock_predictions))

    @staticmethod
    def test_mask(get_join_smoothing_job: SmoothingJob) -> None:
        """Test seam mask in case where tiles don't overlap."""
        smooth = get_join_smoothing_job

        join_index = smooth.tiler.tile_size_h, smooth.tiler.tile_size_w

        # seam should be covered by True
        assert smooth.seam_mask[join_index]

        # non-seam region should be false
        assert not smooth.seam_mask[0, 0]
        assert not smooth.seam_mask[-1, -1]

    @staticmethod
    def test_mask_overlapping(get_ensemble_config: dict, get_batch_predictions: list[ImageBatch]) -> None:
        """Test seam mask in case where tiles overlap."""
        config = copy.deepcopy(get_ensemble_config)
        # tile size = 50, stride = 25 -> overlapping
        config["tiling"]["stride"] = 25
        job_gen = SmoothingJobGenerator(
            accelerator=config["accelerator"],
            tiling_args=config["tiling"],
            data_args=config["data"],
        )
        mock_predictions = copy.deepcopy(get_batch_predictions)
        smooth = next(job_gen.generate_jobs(config["SeamSmoothing"], mock_predictions))

        join_index = smooth.tiler.stride_h, smooth.tiler.stride_w

        # overlap seam should be covered by True
        assert smooth.seam_mask[join_index]
        assert smooth.seam_mask[-join_index[0], -join_index[1]]

        # non-seam region should be false
        assert not smooth.seam_mask[0, 0]
        assert not smooth.seam_mask[-1, -1]

    @staticmethod
    def test_smoothing(get_join_smoothing_job: SmoothingJob, get_batch_predictions: list[ImageBatch]) -> None:
        """Test smoothing job run."""
        original_data = get_batch_predictions
        # fixture makes a copy of data
        smooth = get_join_smoothing_job

        # take first batch
        smoothed = smooth.run()[0]
        join_index = smooth.tiler.tile_size_h, smooth.tiler.tile_size_w

        # join sections should be processed
        assert not smoothed.anomaly_map[:, join_index].equal(original_data[0].anomaly_map[:, join_index])

        # non-join section shouldn't be changed
        assert smoothed.anomaly_map[:, 0, 0].equal(original_data[0].anomaly_map[:, 0, 0])


def test_normalization(get_batch_predictions: list[ImageBatch], project_path: Path) -> None:
    """Test normalization step."""
    original_predictions = copy.deepcopy(get_batch_predictions)

    for batch in original_predictions:
        batch.anomaly_map *= 100
        batch.pred_score *= 100

    # # get and save stats using stats job on predictions
    stats_job_generator = StatisticsJobGenerator(project_path)
    stats_job = next(stats_job_generator.generate_jobs(prev_stage_result=original_predictions))
    stats = stats_job.run()
    stats_job.save(stats)

    # normalize predictions based on obtained stats
    norm_job_generator = NormalizationJobGenerator(root_dir=project_path)
    # copy as this changes preds
    norm_job = next(norm_job_generator.generate_jobs(prev_stage_result=original_predictions))
    normalized_predictions = norm_job.run()

    for batch in normalized_predictions:
        assert (batch.anomaly_map >= 0).all()
        assert (batch.anomaly_map <= 1).all()

        assert (batch.pred_score >= 0).all()
        assert (batch.pred_score <= 1).all()


class TestThresholding:
    """Test tiled ensemble thresholding stage."""

    @pytest.fixture(scope="class")
    @staticmethod
    def get_threshold_job(get_mock_stats_dir: Path) -> callable:
        """Return a function that takes prediction data and runs threshold job."""
        thresh_job_generator = ThresholdingJobGenerator(
            root_dir=get_mock_stats_dir,
            normalization_stage=NormalizationStage.IMAGE,
        )

        def thresh_helper(preds: dict) -> list | None:
            thresh_job = next(thresh_job_generator.generate_jobs(prev_stage_result=preds))
            return thresh_job.run()

        return thresh_helper

    @staticmethod
    def test_score_threshold(get_threshold_job: callable) -> None:
        """Test anomaly score thresholding."""
        thresholding = get_threshold_job

        data = [ImageBatch(image=torch.rand(1, 3, 10, 10), pred_score=torch.tensor([0.7, 0.8, 0.1, 0.33, 0.5]))]

        thresholded = thresholding(data)[0]

        assert thresholded.pred_label.equal(torch.tensor([True, True, False, False, True]))

    @staticmethod
    def test_anomap_threshold(get_threshold_job: callable) -> None:
        """Test anomaly map thresholding."""
        thresholding = get_threshold_job

        data = [
            ImageBatch(
                image=torch.rand(1, 3, 10, 10),
                pred_score=torch.tensor([0.7, 0.8, 0.1, 0.33, 0.5]),
                anomaly_map=torch.tensor([[0.7, 0.8, 0.1], [0.33, 0.5, 0.1]]),
            ),
        ]

        thresholded = thresholding(data)[0]

        assert thresholded.pred_mask.equal(torch.tensor([[[True, True, False], [False, True, False]]]))
