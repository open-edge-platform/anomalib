# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning callback for sending progress to the frontend via the Plugin API."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from lightning.pytorch.callbacks import Callback

from pydantic_models.job import JobStage

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer

logger = logging.getLogger(__name__)


class ProgressSyncParams:
    def __init__(self) -> None:
        self._progress = 0
        self._stage = JobStage.IDLE
        self._lock = threading.Lock()
        self.cancel_training_event = threading.Event()

    @property
    def stage(self) -> JobStage:
        with self._lock:
            return self._stage

    @stage.setter
    def stage(self, stage: JobStage) -> None:
        with self._lock:
            self._stage = stage
        logger.debug("Stage updated: %s", stage)

    @property
    def progress(self) -> int:
        with self._lock:
            return self._progress

    @progress.setter
    def progress(self, progress: int) -> None:
        with self._lock:
            self._progress = progress
        logger.debug("Progress updated: %s", progress)

    def set_cancel_training_event(self) -> None:
        with self._lock:
            self.cancel_training_event.set()
        logger.debug("Set cancel training event")


class GetiInspectProgressCallback(Callback):
    """Callback for displaying training/validation/testing progress in the Geti Inspect UI.

    This callback sends progress events through a multiprocessing queue that the
    main process polls and broadcasts via WebSocket to connected frontend clients.

    Args:
        synchronization_parameters: Parameters for synchronization between the main process and the training process

    Example:
        trainer = Trainer(callbacks=[GetiInspectProgressCallback(synchronization_parameters=ProgressSyncParams())])
    """

    def __init__(self, synchronization_parameters: ProgressSyncParams) -> None:
        """Initialize the callback with synchronization parameters.
        Args:
            synchronization_parameters: Parameters for synchronization between the main process and the training process
        """
        self.synchronization_parameters = synchronization_parameters

    def _check_cancel_training(self, trainer: Trainer) -> None:
        """Check if training should be canceled."""
        if self.synchronization_parameters.cancel_training_event.is_set():
            trainer.should_stop = True

    def _send_progress(self, progress: float, stage: JobStage) -> None:
        """Send progress update to frontend via event queue.
        Puts a generic event message into the multiprocessing queue which will
        be picked up by the main process and broadcast via WebSocket.
        Args:
            progress: Progress value between 0.0 and 1.0
            stage: The current training stage
        """
        # Convert progress to percentage (0-100)
        progress_percent = int(progress * 100)

        try:
            logger.debug("Sent progress: %s - %d%%", stage, progress_percent)
            self.synchronization_parameters.progress = progress_percent
            self.synchronization_parameters.stage = stage
        except Exception as e:
            logger.warning("Failed to send progress to event queue: %s", e)

    # Training callbacks
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training starts."""
        self._send_progress(0, JobStage.TRAINING)
        self._check_cancel_training(trainer)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        """Called when a training batch starts."""
        self._check_cancel_training(trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when a training epoch ends."""
        progress = (trainer.current_epoch + 1) / trainer.max_epochs
        self._send_progress(progress, JobStage.TRAINING)
        self._check_cancel_training(trainer)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training ends."""
        self._send_progress(1.0, JobStage.TRAINING)
        self._check_cancel_training(trainer)

    # Validation callbacks
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when validation starts."""
        self._check_cancel_training(trainer)

    def on_validation_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Called when a validation batch starts."""
        self._check_cancel_training(trainer)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when a validation epoch ends."""
        self._check_cancel_training(trainer)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when validation ends."""
        self._check_cancel_training(trainer)

    # Test callbacks
    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when testing starts."""
        self._send_progress(0, JobStage.TESTING)
        self._check_cancel_training(trainer)

    def on_test_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Called when a test batch starts."""
        self._check_cancel_training(trainer)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when a test epoch ends."""
        progress = (trainer.current_epoch + 1) / trainer.max_epochs if trainer.max_epochs else 0.5
        self._send_progress(progress, JobStage.TESTING)
        self._check_cancel_training(trainer)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when testing ends."""
        self._send_progress(1.0, JobStage.TESTING)
        self._check_cancel_training(trainer)

    # Predict callbacks
    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when prediction starts."""
        self._send_progress(0, JobStage.PREDICTING)
        self._check_cancel_training(trainer)

    def on_predict_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Called when a prediction batch starts."""
        self._check_cancel_training(trainer)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when a prediction epoch ends."""
        progress = (trainer.current_epoch + 1) / trainer.max_epochs if trainer.max_epochs else 0.5
        self._send_progress(progress, JobStage.PREDICTING)
        self._check_cancel_training(trainer)

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when prediction ends."""
        self._send_progress(1.0, JobStage.PREDICTING)
        self._check_cancel_training(trainer)
