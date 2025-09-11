# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import get_model

from db.engine import get_async_db_session_ctx
from models import JobStatus, Model
from repositories.binary_repo import ImageBinaryRepository, ModelBinaryRepository
from services import ModelService
from services.job_service import JobService
from utils import is_platform_darwin

logger = logging.getLogger(__name__)


class TrainingService:
    """
    Service for managing model training jobs.

    Handles the complete training pipeline including job fetching, model training,
    status updates, and error handling. Currently uses asyncio.to_thread for
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
        async with get_async_db_session_ctx() as session:
            job_service = JobService(session)
            job = await job_service.get_pending_train_job()
            if job is None:
                logger.info("No pending training job")
                return None

            project_id = job.project_id
            model_name = job.payload.get("model_name")
            model_service = ModelService(session)
            model = Model(
                project_id=project_id,
                name=model_name,
            )
            logger.info(f"Training model `{model_name}` for job `{job.id}`")

            try:
                # Use asyncio.to_thread to keep event loop responsive
                # TODO: Consider ProcessPoolExecutor for true parallelism with multiple jobs
                trained_model = await asyncio.to_thread(cls._train_model, model)
                if trained_model is None:
                    raise ValueError("Training failed - model is None")

                await job_service.update_job_status(
                    job_id=job.id, status=JobStatus.COMPLETED, message="Training completed successfully"
                )
                return await model_service.create_model(trained_model)
            except Exception as e:
                logger.exception("Failed to train pending training job: %s", e)
                await job_service.update_job_status(
                    job_id=job.id, status=JobStatus.FAILED, message=f"Failed with exception: {str(e)}"
                )
                if model.export_path:
                    logger.warning("Deleting partially created model with id: %s", model.id)
                    model_binary_repo = ModelBinaryRepository(project_id=project_id, model_id=model.id)
                    await model_binary_repo.delete_model_folder()
                    await model_service.delete_model(project_id=project_id, model_id=model.id)
                raise e

    @staticmethod
    def _train_model(model: Model) -> Model | None:
        """
        Execute CPU-intensive model training using anomalib.

        This synchronous function runs in a separate thread via asyncio.to_thread
        to prevent blocking the event loop. Sets up the anomalib model, trains it
        on the dataset, and exports it in OpenVINO format.

        Args:
            model: Model object with training configuration

        Returns:
            Model: Trained model with updated export_path and is_ready=True
        """
        model_binary_repo = ModelBinaryRepository(project_id=model.project_id, model_id=model.id)
        image_binary_repo = ImageBinaryRepository(project_id=model.project_id)
        image_folder_path = image_binary_repo.project_folder_path
        model.export_path = model_binary_repo.model_folder_path
        name = f"{model.project_id}-{model.name}"

        # Configure datamodule for anomalib training
        datamodule = Folder(
            name=name,
            normal_dir=image_folder_path,
            test_split_mode=TestSplitMode.SYNTHETIC,
            num_workers=0 if is_platform_darwin() else 8,
        )
        logger.info(f"Training from image folder: {image_folder_path} to model folder: {model.export_path}")

        # Initialize anomalib model and engine
        anomalib_model = get_model(model=model.name)
        engine = Engine(default_root_dir=model.export_path)

        # Execute training and export
        export_format = ExportType.OPENVINO
        engine.train(model=anomalib_model, datamodule=datamodule)
        export_path = engine.export(
            model=anomalib_model,
            export_type=export_format,
            export_root=model_binary_repo.model_folder_path,
        )
        logger.info(f"Exporting model to {export_path}")

        model.is_ready = True
        return model
