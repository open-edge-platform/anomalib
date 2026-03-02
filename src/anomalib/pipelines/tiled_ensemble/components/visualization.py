# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiled ensemble - visualization job."""

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

from tqdm import tqdm

from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, RUN_RESULTS
from anomalib.utils.path import generate_output_filename
from anomalib.visualization import visualize_image_item
from anomalib.visualization.image.item_visualizer import (
    DEFAULT_FIELDS_CONFIG,
    DEFAULT_OVERLAY_FIELDS_CONFIG,
    DEFAULT_TEXT_CONFIG,
)

logger = logging.getLogger(__name__)


class VisualizationJob(Job):
    """Job for visualization of predictions.

    Args:
        predictions (list[Any]): list of image-level predictions.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        data_args (Dict): data args used to get data name and category name.
    """

    name = "Visualize"

    def __init__(self, predictions: list[Any], root_dir: Path, data_args: dict) -> None:
        super().__init__()
        self.predictions = predictions
        self.root_dir = root_dir / "images"

        self.fields = ["image", "gt_mask"]
        self.overlay_fields = [("image", ["anomaly_map"]), ("image", ["pred_mask"])]
        self.field_size = (256, 256)
        self.fields_config = DEFAULT_FIELDS_CONFIG
        self.overlay_fields_config = DEFAULT_OVERLAY_FIELDS_CONFIG
        self.text_config = DEFAULT_TEXT_CONFIG

        self.dataset_name = data_args["init_args"].get("name", None)
        if self.dataset_name is None:
            # if not specified, take class name
            self.dataset_name = data_args["class_path"].split(".")[-1]
        self.category = data_args["init_args"].get("category", "")

    def run(self, task_id: int | None = None) -> list[Any]:
        """Run job that visualizes all prediction data.

        Args:
            task_id: Not used in this case.

        Returns:
            list[Any]: Unchanged predictions.
        """
        del task_id  # not needed here

        logger.info("Starting visualization.")

        for batch in tqdm(self.predictions, desc="Visualizing"):
            for item in batch:
                image = visualize_image_item(
                    item,
                    fields=self.fields,
                    overlay_fields=self.overlay_fields,
                    field_size=self.field_size,
                    fields_config=self.fields_config,
                    overlay_fields_config=self.overlay_fields_config,
                    text_config=self.text_config,
                )

                if image is not None:
                    # Get the dataset name and category to save the image
                    filename = generate_output_filename(
                        input_path=item.image_path or "",
                        output_path=self.root_dir,
                        dataset_name=self.dataset_name,
                        category=self.category,
                    )

                    # Save the image to the specified filename
                    image.save(filename)

        return self.predictions

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: Unchanged list of predictions.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """This job doesn't save anything."""


class VisualizationJobGenerator(JobGenerator):
    """Generate VisualizationJob.

    Args:
        root_dir (Path): Root directory where images will be saved (root/images).
    """

    def __init__(self, root_dir: Path, data_args: dict) -> None:
        self.root_dir = root_dir
        self.data_args = data_args

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return VisualizationJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: list[Any] | None = None,
    ) -> Generator[VisualizationJob, None, None]:
        """Return a generator producing a single visualization job.

        Args:
            args: Ensemble run args.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[VisualizationJob, None, None]: VisualizationJob generator
        """
        del args  # args not used here

        if prev_stage_result is not None:
            yield VisualizationJob(prev_stage_result, self.root_dir, data_args=self.data_args)
        else:
            msg = "Visualization job requires tile level predictions from previous step."
            raise ValueError(msg)
