# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmarking job for evaluating model performance.

This module provides functionality for running individual benchmarking jobs that
evaluate model performance on specific datasets. Each job runs a model on a dataset
and collects performance metrics.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Padim
    >>> from anomalib.pipelines.benchmark.job import BenchmarkJob

    >>> # Initialize model, datamodule and job
    >>> model = Padim()
    >>> datamodule = MVTecAD(category="bottle")
    >>> job = BenchmarkJob(
    ...     accelerator="gpu",
    ...     model=model,
    ...     datamodule=datamodule,
    ...     seed=42,
    ...     flat_cfg={"model.name": "padim"}
    ... )

    >>> # Run the benchmark job
    >>> results = job.run()

The job executes model training and evaluation, collecting metrics like accuracy,
F1-score, and inference time. Results are returned in a standardized format for
comparison across different model-dataset combinations.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
from lightning import seed_everything
from rich.console import Console
from rich.table import Table

from anomalib.data import AnomalibDataModule
from anomalib.engine import Engine
from anomalib.models import AnomalibModule
from anomalib.pipelines.components import Job
from anomalib.utils.logging import hide_output

logger = logging.getLogger(__name__)


class BenchmarkJob(Job):
    """Benchmarking job for evaluating anomaly detection models.

    This class implements a benchmarking job that evaluates model performance by
    training and testing on a given dataset. It collects metrics like accuracy,
    F1-score, and timing information.

    Args:
        accelerator (str): Type of accelerator to use for computation (e.g.
            ``"cpu"``, ``"gpu"``).
        model (AnomalibModule): Anomaly detection model instance to benchmark.
        datamodule (AnomalibDataModule): Data module providing the dataset.
        seed (int): Random seed for reproducibility.
        flat_cfg (dict): Flattened configuration dictionary with dotted keys.

    Example:
        >>> from anomalib.data import MVTecAD
        >>> from anomalib.models import Padim
        >>> from anomalib.pipelines.benchmark.job import BenchmarkJob

        >>> # Initialize model, datamodule and job
        >>> model = Padim()
        >>> datamodule = MVTecAD(category="bottle")
        >>> job = BenchmarkJob(
        ...     accelerator="gpu",
        ...     model=model,
        ...     datamodule=datamodule,
        ...     seed=42,
        ...     flat_cfg={"model.name": "padim"}
        ... )

        >>> # Run the benchmark job
        >>> results = job.run()

    The job executes model training and evaluation, collecting metrics like
    accuracy, F1-score, and inference time. Results are returned in a standardized
    format for comparison across different model-dataset combinations.
    """

    name = "benchmark"

    def __init__(
        self,
        accelerator: str,
        model: AnomalibModule,
        datamodule: AnomalibDataModule,
        seed: int,
        flat_cfg: dict,
    ) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.model = model
        self.datamodule = datamodule
        self.seed = seed
        self.flat_cfg = flat_cfg

    @hide_output
    def run(
        self,
        task_id: int | None = None,
    ) -> dict[str, Any]:
        """Run the benchmark job.

        This method executes the full benchmarking pipeline including model
        training and testing. It measures execution time for different stages and
        collects performance metrics.

        Args:
            task_id (int | None, optional): ID of the task when running in
                distributed mode. When provided, the job will use the specified
                device. Defaults to ``None``.

        Returns:
            dict[str, Any]: Dictionary containing benchmark results including:
                - Timing information (job, fit and test duration)
                - Model configuration
                - Performance metrics from testing
        """
        # Config knobs for optional inference micro-benchmarking.
        # Note: `flat_cfg` is produced by flattening the `benchmark:` config section,
        # so users typically specify `inference.*` under `benchmark:`.
        inference_bench_enabled = bool(
            self.flat_cfg.get(
                "benchmark.inference.enabled",
                self.flat_cfg.get("inference.enabled", False),
            ),
        )
        inference_warmup_runs = int(
            self.flat_cfg.get(
                "benchmark.inference.warmup_runs",
                self.flat_cfg.get("inference.warmup_runs", 10),
            ),
        )
        inference_runs = int(
            self.flat_cfg.get(
                "benchmark.inference.runs",
                self.flat_cfg.get("inference.runs", 50),
            ),
        )
        inference_batch_size = self.flat_cfg.get(
            "benchmark.inference.batch_size",
            self.flat_cfg.get("inference.batch_size", None),
        )

        job_start_time = time.time()
        devices: str | list[int] = "auto"
        if task_id is not None:
            devices = [task_id]
            logger.info(f"Running job {self.model.__class__.__name__} with device {task_id}")
        with TemporaryDirectory() as temp_dir:
            seed_everything(self.seed)
            engine = Engine(
                accelerator=self.accelerator,
                devices=devices,
                default_root_dir=temp_dir,
            )
            fit_start_time = time.time()
            engine.fit(self.model, self.datamodule)

            # Optional micro-benchmark: torch-only forward-pass throughput.
            # This runs on the same device as the trained model and does not affect metrics.
            inference_bench: dict[str, Any] = {}
            if inference_bench_enabled:
                inference_bench = self._benchmark_inference_throughput(
                    warmup_runs=inference_warmup_runs,
                    runs=inference_runs,
                    batch_size=inference_batch_size,
                )

            test_start_time = time.time()
            test_results = engine.test(self.model, self.datamodule)
        job_end_time = time.time()
        durations = {
            "job_duration": job_end_time - job_start_time,
            "fit_duration": test_start_time - fit_start_time,
            "test_duration": job_end_time - test_start_time,
        }

        # Restore throughput in benchmark outputs.
        # https://github.com/open-edge-platform/anomalib/issues/2054
        test_loader_info = self._get_test_loader_info()
        throughput: dict[str, Any] = {}
        if test_loader_info["num_images"] is not None and durations["test_duration"] > 0:
            throughput = {
                "test_num_images": test_loader_info["num_images"],
                "test_batch_size": test_loader_info["batch_size"],
                "test_throughput_fps": test_loader_info["num_images"] / durations["test_duration"],
            }

        output = {
            "accelerator": self.accelerator,
            **durations,
            **throughput,
            **inference_bench,
            **self.flat_cfg,
            **test_results[0],
        }
        logger.info(f"Completed with result {output}")
        return output

    def _get_test_loader_info(self) -> dict[str, int | None]:
        """Best-effort test loader metadata.

        Returns:
            dict[str, int | None]: Keys:
                - num_images: Total number of images in the test dataset, if available.
                - batch_size: Batch size of the test dataloader, if available.
        """
        try:
            dataloaders = self.datamodule.test_dataloader()
        except Exception:  # noqa: BLE001 - best-effort only
            return {"num_images": None, "batch_size": None}

        dataloader = dataloaders[0] if isinstance(dataloaders, (list, tuple)) else dataloaders
        num_images: int | None
        batch_size: int | None
        try:
            num_images = len(dataloader.dataset)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - best-effort only
            num_images = None
        batch_size = getattr(dataloader, "batch_size", None)
        return {"num_images": num_images, "batch_size": batch_size}

    def _benchmark_inference_throughput(  # noqa: C901
        self,
        warmup_runs: int,
        runs: int,
        batch_size: int | None,
    ) -> dict[str, Any]:
        """Benchmark torch forward-pass throughput on the test dataloader.

        This measures the runtime of the model's underlying torch module
        (`AnomalibModule.model`) on batches from the test dataloader.

        Args:
            warmup_runs: Number of warmup iterations.
            runs: Number of measured iterations.
            batch_size: Optional override for dataloader batch size.

        Returns:
            dict[str, Any]: Benchmark metrics. Returns an empty dict if unavailable.
        """
        try:
            import torch
            from torch.utils.data import DataLoader

            dataloaders = self.datamodule.test_dataloader()
            dataloader = dataloaders[0] if isinstance(dataloaders, (list, tuple)) else dataloaders
            if not hasattr(dataloader, "dataset"):
                return {}

            if batch_size is not None and getattr(dataloader, "batch_size", None) != batch_size:
                # Re-create a DataLoader with a different batch size while preserving common settings.
                # This intentionally does not try to preserve complex samplers.
                kwargs: dict[str, Any] = {
                    "dataset": dataloader.dataset,
                    "batch_size": batch_size,
                    "shuffle": False,
                    "num_workers": getattr(dataloader, "num_workers", 0),
                    "pin_memory": getattr(dataloader, "pin_memory", False),
                    "drop_last": getattr(dataloader, "drop_last", False),
                    "collate_fn": getattr(dataloader, "collate_fn", None),
                }
                for key in ("timeout", "worker_init_fn", "prefetch_factor", "persistent_workers"):
                    if hasattr(dataloader, key):
                        kwargs[key] = getattr(dataloader, key)
                dataloader = DataLoader(**kwargs)

            device = getattr(self.model, "device", None)
            if device is None:
                try:
                    device = next(self.model.parameters()).device
                except Exception:  # noqa: BLE001
                    device = torch.device("cpu")

            core_model = getattr(self.model, "model", None)
            if core_model is None:
                return {}
            core_model.eval()

            def _get_images(batch: Any) -> Any:  # noqa: ANN401
                if hasattr(batch, "image"):
                    return batch.image
                if isinstance(batch, dict) and "image" in batch:
                    return batch["image"]
                return None

            def _batch_stream() -> Any:  # noqa: ANN401
                while True:
                    yield from dataloader

            batch_iter = _batch_stream()
            total_measured_images = 0

            def _run(num_iters: int, count_images: bool) -> None:
                nonlocal total_measured_images
                for _ in range(num_iters):
                    batch = next(batch_iter)
                    images = _get_images(batch)
                    if images is None:
                        return
                    images = images.to(device)
                    _ = core_model(images)
                    if count_images:
                        total_measured_images += int(images.shape[0])

            with torch.inference_mode():
                _run(max(0, warmup_runs), count_images=False)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                _run(max(1, runs), count_images=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                total_measured_seconds = time.perf_counter() - start

            if total_measured_images <= 0 or total_measured_seconds <= 0:
                return {}

            return {
                "inference_warmup_runs": warmup_runs,
                "inference_runs": runs,
                "inference_batch_size": getattr(dataloader, "batch_size", None),
                "inference_num_images": total_measured_images,
                "inference_duration": total_measured_seconds,
                "inference_throughput_fps": total_measured_images / total_measured_seconds,
                "inference_seconds_per_image": total_measured_seconds / total_measured_images,
            }
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def collect(results: list[dict[str, Any]]) -> pd.DataFrame:
        """Collect and aggregate results from multiple benchmark runs.

        Args:
            results (list[dict[str, Any]]): List of result dictionaries from
                individual benchmark runs.

        Returns:
            pd.DataFrame: DataFrame containing aggregated results with each row
                representing a benchmark run.
        """
        output: dict[str, Any] = {}
        for key in results[0]:
            output[key] = []
        for result in results:
            for key, value in result.items():
                output[key].append(value)
        return pd.DataFrame(output)

    @staticmethod
    def save(result: pd.DataFrame) -> None:
        """Save benchmark results to CSV file.

        The results are saved in the ``runs/benchmark/YYYY-MM-DD-HH_MM_SS``
        directory. The method also prints a tabular view of the results.

        Args:
            result (pd.DataFrame): DataFrame containing benchmark results to save.
        """
        BenchmarkJob._print_tabular_results(result)
        file_path = Path("runs") / BenchmarkJob.name / datetime.now().strftime("%Y-%m-%d-%H_%M_%S") / "results.csv"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(file_path, index=False)
        logger.info(f"Saved results to {file_path}")

    @staticmethod
    def _print_tabular_results(gathered_result: pd.DataFrame) -> None:
        """Print benchmark results in a formatted table.

        Args:
            gathered_result (pd.DataFrame): DataFrame containing results to
                display.
        """
        if gathered_result is not None:
            console = Console()
            table = Table(title=f"{BenchmarkJob.name} Results", show_header=True, header_style="bold magenta")
            results = gathered_result.to_dict("list")
            for column in results:
                table.add_column(column)
            for row in zip(*results.values(), strict=False):
                table.add_row(*[str(value) for value in row])
            console.print(table)
