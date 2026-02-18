# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import pathlib
from contextlib import redirect_stdout
from typing import Any
from uuid import UUID

from anomalib.data import Folder
from anomalib.data.utils import ValSplitMode
from anomalib.deploy import ExportType
from anomalib.engine import Engine, XPUAccelerator
from anomalib.engine.strategy.xpu_single import SingleXPUStrategy
from anomalib.loggers import AnomalibTensorBoardLogger
from anomalib.metrics import AUROC, F1Score
from anomalib.metrics.evaluator import Evaluator
from anomalib.models import get_model
from lightning.pytorch.callbacks import EarlyStopping
from loguru import logger

from pydantic_models import Job, JobStatus, JobType, Model
from repositories.binary_repo import ModelBinaryRepository
from services import ModelService
from services.dataset_snapshot_service import DatasetSnapshotService
from services.job_service import JobService
from services.system_service import SystemService
from utils.callbacks import AnomalibStudioProgressCallback, ProgressSyncParams


class TrainingService:
    """Service for managing model training jobs.

    Handles the complete training pipeline including job fetching, model training,
    status updates, and error handling. Currently, using asyncio.to_thread for
    CPU-intensive training to maintain event loop responsiveness.

    Note: asyncio.to_thread is used assuming single concurrent training job.
    For true parallelism with multiple training jobs, consider ProcessPoolExecutor.
    """

    @classmethod
    async def train_pending_job(cls) -> Model | None:
        """Process the next pending training job from the queue.

        Fetches a pending job, executes training in a separate thread to maintain
        event loop responsiveness, and updates job status accordingly.

        Returns:
            Model: Trained model if successful, None if no pending jobs
        """
        job_service = JobService()
        job = await job_service.get_pending_train_job()
        if job is None:
            logger.trace("No pending training job")
            return None

        # Run the training job with logging context
        from core.logging.utils import job_logging_ctx  # noqa: PLC0415

        with job_logging_ctx(job_id=str(job.id)):
            return await cls._run_training_job(job, job_service)

    @classmethod
    async def _run_training_job(cls, job: Job, job_service: JobService) -> Model | None:
        # Mark job as running
        await job_service.update_job_status(job_id=job.id, status=JobStatus.RUNNING, message="Training started")
        project_id = job.project_id
        model_name = job.payload.get("model_name")
        device = job.payload.get("device")
        snapshot_id = UUID(snapshot_id_str) if (snapshot_id_str := job.payload.get("dataset_snapshot_id")) else None
        # UI can return None
        max_epochs: int = payload_epochs if (payload_epochs := job.payload.get("max_epochs")) is not None else 200
        if model_name is None:
            raise ValueError(f"Job {job.id} payload must contain 'model_name'")

        synchronization_task: asyncio.Task[None] | None = None
        model: Model | None = None  # Initialize model to None

        try:
            model_service = ModelService()
            snapshot = await DatasetSnapshotService.get_or_create_snapshot(
                project_id=project_id,
                snapshot_id=snapshot_id,
            )
            snapshot_id = snapshot.id

            model = Model(
                project_id=project_id,
                name=str(model_name),
                train_job_id=job.id,
                dataset_snapshot_id=snapshot_id,
            )
            synchronization_parameters = ProgressSyncParams()
            logger.info(f"Training model `{model_name}` for job `{job.id}` using snapshot `{snapshot_id}`")

            synchronization_task = asyncio.create_task(
                cls._sync_progress_with_db(
                    job_service=job_service,
                    job_id=job.id,
                    synchronization_parameters=synchronization_parameters,
                ),
            )

            # Use the context manager from DatasetSnapshotService to prepare data
            async with DatasetSnapshotService.use_snapshot_as_folder(snapshot_id, project_id) as dataset_root:
                # Use asyncio.to_thread to keep event loop responsive
                # TODO: Consider ProcessPoolExecutor for true parallelism with multiple jobs
                trained_model = await asyncio.to_thread(
                    cls._train_model,
                    model=model,
                    device=device,
                    synchronization_parameters=synchronization_parameters,
                    max_epochs=max_epochs,
                    dataset_root=dataset_root,
                )

            if synchronization_parameters.cancel_training_event.is_set():
                await cls._handle_job_cancellation(job_service=job_service, job=job, model=model)
                return None

            if trained_model is None:
                raise ValueError("Training failed - model is None")

            return await model_service.create_model(trained_model)
        except Exception as e:
            logger.error(f"Failed to train pending training job: {e}")
            await job_service.update_job_status(
                job_id=job.id,
                status=JobStatus.FAILED,
                message=f"Failed with exception: {str(e)}",
            )
            if model and model.export_path:
                logger.warning(f"Deleting partially created model with id: {model.id}")
                await model_service.delete_model(project_id=project_id, model_id=model.id)
            raise e
        finally:
            logger.debug("Syncing progress with db stopped")
            if synchronization_task is not None and not synchronization_task.done():
                synchronization_task.cancel()
                try:
                    await synchronization_task
                except asyncio.CancelledError:
                    logger.info("Synchronization task cancelled successfully")
                except Exception as e:
                    logger.error(f"Synchronization task failed with: `{e}`")

            # bookkeeping after training completion
            # update must happen after synchronization task is cancelled to avoid overwriting
            job_ = await job_service.get_job_by_id(job_id=job.id)
            if job_ is not None and job_.is_running:
                logger.success(f"Successfully trained model: `{model_name}`")
                await job_service.update_job_status(
                    job_id=job.id,
                    status=JobStatus.COMPLETED,
                    message="Training completed successfully",
                )
            # Cleanup unused snapshot
            # If training succeeded, model is created and references snapshot, so it won't delete.
            # If training failed (model not created), it will delete if no other model uses it.
            if snapshot_id:
                await DatasetSnapshotService.delete_snapshot_if_unused(snapshot_id=snapshot_id, project_id=project_id)

    @staticmethod
    def _train_model(
        model: Model,
        synchronization_parameters: ProgressSyncParams,
        dataset_root: str,
        max_epochs: int,
        device: str | None = None,
    ) -> Model | None:
        """Execute CPU-intensive model training using anomalib.

        This synchronous function runs in a separate thread via asyncio.to_thread
        to prevent blocking the event loop. Sets up the anomalib model, trains it
        on the dataset, and exports it in OpenVINO format.

        Args:
            model: Model object with training configuration
            synchronization_parameters: Parameters for synchronization between the main process and the training process
            dataset_root: Path to the temporary folder containing the extracted dataset
            device: Device to train on

        Returns:
            Model: Trained model with updated export_path and is_ready=True
        """
        from core.logging import global_log_config  # noqa: PLC0415
        from core.logging.handlers import LoggerStdoutWriter  # noqa: PLC0415

        device = device.lower() if device else None  # anomalib expects lowercase device strings
        if device and not SystemService.is_device_supported_for_training(device):
            raise ValueError(
                f"Device '{device}' is not supported for training. "
                f"Supported devices: {', '.join([device.type for device in SystemService.get_training_devices()])}",
            )

        training_device = device or "auto"
        logger.info(f"Training on device: {training_device}")

        model_binary_repo = ModelBinaryRepository(project_id=model.project_id, model_id=model.id)
        model.export_path = model_binary_repo.model_folder_path
        name = f"{model.project_id}-{model.name}"

        normal_dir = os.path.join(dataset_root, "normal")

        logger.info(f"Training from temp folder: {dataset_root}")
        # TODO: implement Parquet datamodule in anomalib to avoid folder extraction step
        datamodule = Folder(
            name=name,
            normal_dir=normal_dir,
            val_split_mode=ValSplitMode.SYNTHETIC,
        )

        # Initialize anomalib model and engine
        anomalib_model = get_model(
            model=model.name,
            evaluator=Evaluator(
                val_metrics=[AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False)],
                test_metrics=[
                    AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
                    F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
                    AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_", strict=False),
                    F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_", strict=False),
                ],
            ),
        )

        tensorboard = AnomalibTensorBoardLogger(save_dir=global_log_config.tensorboard_log_path, name=name)
        kwargs: dict[str, Any] = {}
        if training_device == "xpu":
            kwargs["strategy"] = SingleXPUStrategy()
            kwargs["accelerator"] = XPUAccelerator()
        else:
            kwargs["accelerator"] = training_device

        engine = Engine(
            default_root_dir=model.export_path,
            logger=[tensorboard],
            devices=1,  # Works for all devices. Single GPU training for now. And for CPU, this means single process
            max_epochs=max_epochs,
            callbacks=[
                AnomalibStudioProgressCallback(synchronization_parameters),
                EarlyStopping(monitor="pixel_AUROC", mode="max", patience=5),
            ],
            **kwargs,
        )

        # Execute training and export
        export_format = ExportType.OPENVINO

        # Capture pytorch stdout logs into logger
        with redirect_stdout(LoggerStdoutWriter()):  # type: ignore[type-var]
            engine.fit(model=anomalib_model, datamodule=datamodule)

        # Find and set threshold metric
        for callback in engine.trainer.callbacks:  # type: ignore[attr-defined]
            if threshold := getattr(callback, "normalized_pixel_threshold", None):
                logger.debug(f"Found pixel threshold set to: {threshold}")
                model.threshold = threshold.item()
                break

        if isinstance(getattr(anomalib_model.model, "backbone", None), str):
            model.backbone = anomalib_model.model.backbone  # type: ignore[assignment] # backbone is str here

        if synchronization_parameters.cancel_training_event.is_set():
            return None

        synchronization_parameters.message = "exporting model"

        export_path = engine.export(
            model=anomalib_model,
            export_type=export_format,
            export_root=model_binary_repo.model_folder_path,
        )
        logger.info(f"Exporting model to {export_path}")

        model.is_ready = True
        model.size = TrainingService._compute_export_size(model.export_path)
        return model

    @staticmethod
    async def _handle_job_cancellation(job_service: JobService, job: Job, model: Model) -> None:
        """Mark job as cancelled and remove partially exported artifacts."""
        logger.info("Training job `%s` cancelled by user", job.id)
        await job_service.update_job_status(
            job_id=job.id,
            status=JobStatus.CANCELED,
            message="Training cancelled by user",
        )

        model_binary_repo = ModelBinaryRepository(project_id=job.project_id, model_id=model.id)
        await model_binary_repo.delete_model_folder()

    @staticmethod
    def _compute_export_size(path: str | None) -> int | None:
        if path is None:
            return None

        try:
            path_obj = pathlib.Path(path)
            if path_obj.is_file():
                return path_obj.stat().st_size
            if not path_obj.is_dir():
                logger.warning(f"Cannot compute export size because `{path}` is not a directory")
                return None
        except OSError as error:
            logger.error(f"Failed to access export path `{path}` while computing size: {error}")
            return None

        def iter_file_sizes():
            for root, _, files in os.walk(path, followlinks=False):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    if pathlib.Path(file_path).is_symlink():
                        continue
                    try:
                        yield pathlib.Path(file_path).stat().st_size
                    except OSError:
                        continue

        return sum(iter_file_sizes())

    @classmethod
    async def _sync_progress_with_db(
        cls,
        job_service: JobService,
        job_id: UUID,
        synchronization_parameters: ProgressSyncParams,
    ) -> None:
        try:
            last_progress = 0
            last_message = ""
            while True:
                progress: int = synchronization_parameters.progress
                message = synchronization_parameters.message
                job = await job_service.get_job_by_id(job_id=job_id)
                if job is None:
                    logger.error(f"Job with id {job_id} not found, stopping progress sync")
                    break
                if job.status == JobStatus.CANCELED:
                    logger.info(f"Job with id {job_id} marked as cancelled, stopping training")
                    synchronization_parameters.cancel_training_event.set()
                    break
                if job.status != JobStatus.RUNNING:
                    logger.info(f"Job status changed to {job.status}, stopping progress sync")
                    break

                if progress != last_progress or message != last_message:
                    logger.trace(f"Syncing progress with db: {progress}% - {message}")
                    await job_service.update_job_progress(
                        job_id=job_id,
                        progress=progress,
                        message=message,
                    )
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.exception("Failed to sync progress with db: %s", e)
            await job_service.update_job_status(job_id=job_id, status=JobStatus.FAILED, message="Training failed")
            raise

    @staticmethod
    async def abort_orphan_jobs() -> None:
        """Abort all running orphan training jobs (that do not belong to any worker).

        This method can be called during application shutdown/setup to ensure that
        any orphan in-progress training jobs are marked as failed.
        """
        query = {"status": JobStatus.RUNNING, "type": JobType.TRAINING}
        running_jobs = JobService.get_job_list_streaming(extra_filters=query)
        async for job in running_jobs:
            logger.warning(f"Aborting orphan training job with id: {job.id}")
            await JobService.update_job_status(
                job_id=job.id,
                status=JobStatus.FAILED,
                message="Job aborted due to application shutdown",
            )
