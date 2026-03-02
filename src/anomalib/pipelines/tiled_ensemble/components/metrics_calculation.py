# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiled ensemble - metrics calculation job."""

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from anomalib.metrics import Evaluator
from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, PREV_STAGE_RESULT, RUN_RESULTS

from .utils import NormalizationStage
from .utils.helper_functions import get_ensemble_model

logger = logging.getLogger(__name__)


class MetricsCalculationJob(Job):
    """Job for image and pixel metrics calculation.

    Args:
        accelerator (str): Accelerator (device) to use.
        predictions (list[Any]): List of batch predictions.
        root_dir (Path): Root directory to save checkpoints, stats and images.
    """

    name = "Metrics"

    def __init__(
        self,
        accelerator: str,
        predictions: list[Any] | None,
        root_dir: Path,
        evaluator: Evaluator,
    ) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.predictions = predictions
        self.root_dir = root_dir
        self.evaluator = evaluator

    def run(self, task_id: int | None = None) -> dict:
        """Run a job that calculates image and pixel level metrics.

        Args:
            task_id: Not used in this case.

        Returns:
            dict[str, float]: Dictionary containing calculated metric values.
        """
        del task_id  # not needed here

        logger.info("Starting metrics calculation.")

        # add predicted data to metrics
        for data in tqdm(self.predictions, desc="Calculating metrics"):
            # on_test_batch_end updates test metrics
            self.evaluator.on_test_batch_end(None, None, None, batch=data, batch_idx=0)

        # compute all metrics on specified accelerator
        metrics_dict = {}
        for metric in self.evaluator.test_metrics:
            metric.to(self.accelerator)
            metrics_dict[metric.name] = metric.compute().item()
            metric.cpu()

        for name, value in metrics_dict.items():
            print(f"{name}: {value:.4f}")

        # save path used in `save` method
        metrics_dict["save_path"] = self.root_dir / "metric_results.csv"

        return metrics_dict

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: list of predictions.
        """
        # take the first element as result is list of dict here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Save metrics values to csv."""
        logger.info("Saving metrics to csv.")

        # get and remove path from stats dict
        results_path: Path = results.pop("save_path")
        results_path.parent.mkdir(parents=True, exist_ok=True)

        df_dict = {k: [v] for k, v in results.items()}
        metrics_df = pd.DataFrame(df_dict)
        metrics_df.to_csv(results_path, index=False)


class MetricsCalculationJobGenerator(JobGenerator):
    """Generate MetricsCalculationJob.

    Args:
        root_dir (Path): Root directory to save checkpoints, stats and images.
    """

    def __init__(
        self,
        accelerator: str,
        root_dir: Path,
        model_args: dict,
    ) -> None:
        self.accelerator = accelerator
        self.root_dir = root_dir
        self.model_args = model_args

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return MetricsCalculationJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: PREV_STAGE_RESULT = None,
    ) -> Generator[MetricsCalculationJob, None, None]:
        """Make a generator that yields a single metrics calculation job.

        Args:
            args: ensemble run config.
            prev_stage_result: ensemble predictions from previous step.

        Returns:
            Generator[MetricsCalculationJob, None, None]: MetricsCalculationJob generator
        """
        del args  # args not used here

        model = get_ensemble_model(self.model_args, normalization_stage=NormalizationStage.IMAGE, input_size=(10, 10))

        if model.evaluator is not None:
            yield MetricsCalculationJob(
                accelerator=self.accelerator,
                predictions=prev_stage_result,
                root_dir=self.root_dir,
                evaluator=model.evaluator,
            )
        else:
            msg = "Model passed to tiled ensemble has no evaluator module which is required to calculate metrics."
            raise RuntimeError(msg)
