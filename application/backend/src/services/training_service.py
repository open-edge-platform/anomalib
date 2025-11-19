# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
from contextlib import redirect_stdout
from uuid import UUID

from anomalib.data import Folder
from anomalib.data.utils import ValSplitMode
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.loggers import AnomalibTensorBoardLogger
from anomalib.models import get_model
from loguru import logger

from pydantic_models import Job, JobStatus, JobType, Model
from repositories.binary_repo import ImageBinaryRepository, ModelBinaryRepository
from services import ModelService
from services.job_service import JobService
from utils.callbacks import GetiInspectProgressCallback, ProgressSyncParams
from utils.devices import Devices
from utils.experiment_loggers import TrackioLogger


class TrainingService:
    """
    Service for managing model training jobs.

    Handles the complete training pipeline including job fetching, model training,
    status updates, and error handling. Currently, using asyncio.to_thread for
    CPU-intensive training to maintain event loop responsiveness.

    Note: asyncio.to_thread is used assuming single concurrent training job.
    For true parallelism with multiple training jobs, consider ProcessPoolExecutor.
    """

    @classmethod
    async def train_pending_job(cls) -> Model | None:
        """
        Process the next pending training job from the queue.

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
        if model_name is None:
            raise ValueError(f"Job {job.id} payload must contain 'model_name'")

        model_service = ModelService()
        model = Model(
            project_id=project_id,
            name=str(model_name),
            train_job_id=job.id,
        )
        synchronization_parameters = ProgressSyncParams()
        logger.info(f"Training model `{model_name}` for job `{job.id}`")

        synchronization_task: asyncio.Task[None] | None = None
        try:
            synchronization_task = asyncio.create_task(
                cls._sync_progress_with_db(
                    job_service=job_service, job_id=job.id, synchronization_parameters=synchronization_parameters
                )
            )
            # Use asyncio.to_thread to keep event loop responsive
            # TODO: Consider ProcessPoolExecutor for true parallelism with multiple jobs
            trained_model = await asyncio.to_thread(
                cls._train_model,
                model=model,
                device=device,
                synchronization_parameters=synchronization_parameters,
            )

            if synchronization_parameters.cancel_training_event.is_set():
                await cls._handle_job_cancellation(job_service=job_service, job=job, model=model)
                return None

            if trained_model is None:
                raise ValueError("Training failed - model is None")

            await job_service.update_job_status(
                job_id=job.id, status=JobStatus.COMPLETED, message="Training completed successfully"
            )
            return await model_service.create_model(trained_model)
        except Exception as e:
            logger.error("Failed to train pending training job: %s", e)
            await job_service.update_job_status(
                job_id=job.id, status=JobStatus.FAILED, message=f"Failed with exception: {str(e)}"
            )
            if model.export_path:
                logger.warning("Deleting partially created model with id: %s", model.id)
                await cls._cleanup_partial_model(
                    job=job,
                    model=model,
                    delete_model_record=True,
                    model_service=model_service,
                )
            raise e
        finally:
            logger.debug("Syncing progress with db stopped")
            if synchronization_task is not None and not synchronization_task.done():
                synchronization_task.cancel()

    @staticmethod
    def _train_model(
        model: Model, synchronization_parameters: ProgressSyncParams, device: str | None = None
    ) -> Model | None:
        """
        Execute CPU-intensive model training using anomalib.

        This synchronous function runs in a separate thread via asyncio.to_thread
        to prevent blocking the event loop. Sets up the anomalib model, trains it
        on the dataset, and exports it in OpenVINO format.

        Args:
            model: Model object with training configuration
            synchronization_parameters: Parameters for synchronization between the main process and the training process
            device: Device to train on

        Returns:
            Model: Trained model with updated export_path and is_ready=True
        """
        from core.logging import global_log_config  # noqa: PLC0415
        from core.logging.handlers import LoggerStdoutWriter  # noqa: PLC0415

        if device and not Devices.is_device_supported_for_training(device):
            raise ValueError(
                f"Device '{device}' is not supported for training. "
                f"Supported devices: {', '.join(Devices.training_devices())}"
            )

        training_device = device or "auto"
        logger.info(f"Training on device: {training_device}")

        model_binary_repo = ModelBinaryRepository(project_id=model.project_id, model_id=model.id)
        image_binary_repo = ImageBinaryRepository(project_id=model.project_id)
        image_folder_path = image_binary_repo.project_folder_path
        model.export_path = model_binary_repo.model_folder_path
        name = f"{model.project_id}-{model.name}"

        # Configure datamodule for anomalib training
        datamodule = Folder(
            name=name,
            normal_dir=image_folder_path,
            val_split_mode=ValSplitMode.SYNTHETIC,
        )
        logger.info(f"Training from image folder: {image_folder_path} to model folder: {model.export_path}")

        # Initialize anomalib model and engine
        anomalib_model = get_model(model=model.name)

        trackio = TrackioLogger(project=str(model.project_id), name=model.name)
        tensorboard = AnomalibTensorBoardLogger(save_dir=global_log_config.tensorboard_log_path, name=name)
        engine = Engine(
            default_root_dir=model.export_path,
            logger=[trackio, tensorboard],
            devices=[0],  # Only single GPU training is supported for now
            max_epochs=10,
            callbacks=[GetiInspectProgressCallback(synchronization_parameters)],
            accelerator=training_device,
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

        if synchronization_parameters.cancel_training_event.is_set():
            return None

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
        await TrainingService._cleanup_partial_model(job=job, model=model, delete_model_record=False)

    @staticmethod
    async def _cleanup_partial_model(
        *,
        job: Job,
        model: Model,
        delete_model_record: bool,
        model_service: ModelService | None = None,
    ) -> None:
        """Remove partially exported artifacts and optionally delete model record."""
        if not model.export_path:
            return

        model_binary_repo = ModelBinaryRepository(project_id=job.project_id, model_id=model.id)
        await model_binary_repo.delete_model_folder()

        if delete_model_record:
            service = model_service or ModelService()
            await service.delete_model(project_id=job.project_id, model_id=model.id)

    @staticmethod
    def _compute_export_size(path: str | None) -> int | None:
        if path is None:
            return None

        try:
            if os.path.isfile(path):
                return os.path.getsize(path)
            if not os.path.isdir(path):
                logger.warning(f"Cannot compute export size because `{path}` is not a directory")
                return None
        except OSError as error:
            logger.error(f"Failed to access export path `{path}` while computing size: {error}")
            return None

        def iter_file_sizes():
            for root, _, files in os.walk(path, followlinks=False):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    if os.path.islink(file_path):
                        continue
                    try:
                        yield os.path.getsize(file_path)
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
            while True:
                progress: int = synchronization_parameters.progress
                message = synchronization_parameters.message
                if not await job_service.is_job_still_running(job_id=job_id):
                    logger.debug("Job cancelled, stopping progress sync")
                    synchronization_parameters.set_cancel_training_event()
                    break
                logger.debug(f"Syncing progress with db: {progress}% - {message}")
                await job_service.update_job_status(
                    job_id=job_id, status=JobStatus.RUNNING, progress=progress, message=message
                )
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.exception("Failed to sync progress with db: %s", e)
            await job_service.update_job_status(job_id=job_id, status=JobStatus.FAILED, message="Training failed")
            raise

    @staticmethod
    async def abort_orphan_jobs() -> None:
        """
        Abort all running orphan training jobs (that do not belong to any worker).

        This method can be called during application shutdown/setup to ensure that
        any orphan in-progress training jobs are marked as failed.
        """
        query = {"status": JobStatus.RUNNING, "type": JobType.TRAINING}
        running_jobs = await JobService.get_job_list(extra_filters=query)
        for job in running_jobs.jobs:
            logger.warning(f"Aborting orphan training job with id: {job.id}")
            await JobService.update_job_status(
                job_id=job.id,
                status=JobStatus.FAILED,
                message="Job aborted due to application shutdown",
            )
