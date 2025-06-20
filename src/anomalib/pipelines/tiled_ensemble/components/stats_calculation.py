# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiled ensemble - post-processing statistics calculation job."""

import json
import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

from tqdm import tqdm

from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, RUN_RESULTS
from anomalib.post_processing import PostProcessor

logger = logging.getLogger(__name__)


class StatisticsJob(Job):
    """Job for calculating min, max and threshold statistics for post-processing.

    Args:
        predictions (list[Any]): List of image-level predictions.
        root_dir (Path): Root directory to save checkpoints, stats and images.
    """

    name = "Stats"

    def __init__(
        self,
        predictions: list[Any] | None,
        root_dir: Path,
    ) -> None:
        super().__init__()
        self.predictions = predictions
        self.root_dir = root_dir

    def run(self, task_id: int | None = None) -> dict:
        """Run job that calculates statistics needed in post-processing steps.

        Args:
            task_id: Not used in this case

        Returns:
            dict: Statistics dict with min, max and threshold values.
        """
        del task_id  # not needed here

        post_processor = PostProcessor()

        logger.info("Starting post-processing statistics calculation.")

        for data in tqdm(self.predictions, desc="Stats calculation"):
            # update minmax and thresholds
            post_processor.on_validation_batch_end(None, None, outputs=data)

        post_processor.on_validation_epoch_end(None, None)

        # return stats with save path that is later used to save statistics.
        return {
            "minmax": {
                "pred_score": {
                    "min": post_processor.image_min.item(),
                    "max": post_processor.image_max.item(),
                },
                "anomaly_map": {
                    "min": post_processor.pixel_min.item(),
                    "max": post_processor.pixel_max.item(),
                },
            },
            "image_threshold": post_processor.image_threshold.item(),
            "pixel_threshold": post_processor.pixel_threshold.item(),
            "save_path": (self.root_dir / "weights" / "lightning" / "stats.json"),
        }

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            dict: statistics dictionary.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Save statistics to file system."""
        # get and remove path from stats dict
        stats_path: Path = results.pop("save_path")
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        # save statistics next to weights
        with stats_path.open("w", encoding="utf-8") as stats_file:
            json.dump(results, stats_file, ensure_ascii=False, indent=4)


class StatisticsJobGenerator(JobGenerator):
    """Generate StatisticsJob.

    Args:
        root_dir (Path): Root directory where statistics file will be saved (in weights folder).
    """

    def __init__(
        self,
        root_dir: Path,
    ) -> None:
        self.root_dir = root_dir

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return StatisticsJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: list[Any] | None = None,
    ) -> Generator[StatisticsJob, None, None]:
        """Return a generator producing a single stats calculating job.

        Args:
            args: Not used here.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[StatisticsJob, None, None]: StatisticsJob generator.
        """
        del args  # not needed here

        yield StatisticsJob(
            predictions=prev_stage_result,
            root_dir=self.root_dir,
        )
