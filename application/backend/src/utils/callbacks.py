# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning callback for sending progress to the frontend via the Plugin API."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from lightning.pytorch.callbacks import Callback
from loguru import logger

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


class ProgressSyncParams:
    def __init__(self) -> None:
        self._progress = 0
        self._message: str = "Initializing"
        self._lock = threading.Lock()
        self.cancel_training_event = threading.Event()

    @property
    def message(self) -> str:
        with self._lock:
            return self._message

    @message.setter
    def message(self, stage: str) -> None:
        with self._lock:
            self._message = f"Stage: {stage}"
        logger.debug(f"Message updated: {self._message}")

    @property
    def progress(self) -> int:
        with self._lock:
            return self._progress

    @progress.setter
    def progress(self, progress: int) -> None:
        with self._lock:
            self._progress = progress
        logger.debug(f"Progress updated: {progress}")

    def set_cancel_training_event(self) -> None:
        with self._lock:
            self.cancel_training_event.set()
        logger.debug("Set cancel training event")


class AnomalibStudioProgressCallback(Callback):
    """Callback for displaying training/validation/testing progress in the Anomalib Studio UI.

    This callback sends progress events through a multiprocessing queue that the
    main process polls and broadcasts via WebSocket to connected frontend clients.

    Args:
        synchronization_parameters: Parameters for synchronization between the main process and the training process

    Example:
        trainer = Trainer(callbacks=[AnomalibStudioProgressCallback(synchronization_parameters=ProgressSyncParams())])
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

    def _send_progress(self, progress: float, message: str) -> None:
        """Send progress update to frontend via event queue.
        Puts a generic event message into the multiprocessing queue which will
        be picked up by the main process and broadcast via WebSocket.
        Args:
            progress: Progress value between 0.0 and 1.0
            message: The current training message
        """
        # Convert progress to percentage (0-100)
        progress_percent = int(progress * 100)

        try:
            logger.debug(f"Sent progress: {message} - {progress_percent}%")
            self.synchronization_parameters.progress = progress_percent
            self.synchronization_parameters.message = message
        except Exception as e:
            logger.warning(f"Failed to send progress to event queue: {e}")

    # Training callbacks
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training starts."""
        del pl_module  # unused
        if trainer.state.stage is not None:
            self._send_progress(0, trainer.state.stage.value)
        else:
            self._send_progress(0, "Training started")
        self._check_cancel_training(trainer)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        """Called when a training batch starts."""
        del pl_module, batch, batch_idx  # unused
        self._check_cancel_training(trainer)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when a training batch ends. Sends granular progress updates within each epoch."""
        del pl_module, outputs, batch  # unused
        if trainer.state.stage is not None and trainer.max_epochs is not None and trainer.max_epochs > 0:
            total_batches = trainer.num_training_batches
            if total_batches and total_batches > 0:
                epoch_progress = trainer.current_epoch / trainer.max_epochs
                batch_progress = (batch_idx + 1) / total_batches / trainer.max_epochs
                progress = epoch_progress + batch_progress
                self._send_progress(progress, trainer.state.stage.value)
        self._check_cancel_training(trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when a training epoch ends."""
        del pl_module  # unused
        # If max_epochs is not available, set progress to 0.5
        if trainer.state.stage is not None:
            progress = (
                (trainer.current_epoch + 1) / trainer.max_epochs
                if (trainer.max_epochs is not None and trainer.max_epochs > 0)
                else 0.5
            )
            self._send_progress(progress, trainer.state.stage.value)
        self._check_cancel_training(trainer)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training ends."""
        del pl_module  # unused
        if trainer.state.stage is not None:
            self._send_progress(1.0, trainer.state.stage.value)
        self._check_cancel_training(trainer)

    # Validation callbacks
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when validation starts."""
        del pl_module  # unused
        self._check_cancel_training(trainer)

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when a validation batch starts."""
        del pl_module, batch, batch_idx, dataloader_idx  # unused
        self._check_cancel_training(trainer)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when a validation epoch ends."""
        del pl_module  # unused
        self._check_cancel_training(trainer)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when validation ends."""
        del pl_module  # unused
        self._check_cancel_training(trainer)

    # Test callbacks
    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when testing starts."""
        del pl_module  # unused
        if trainer.state.stage is not None:
            self._send_progress(0, trainer.state.stage.value)
        else:
            self._send_progress(0, "Testing started")
        self._check_cancel_training(trainer)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when a test batch starts."""
        del pl_module, batch, batch_idx, dataloader_idx  # unused
        self._check_cancel_training(trainer)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when a test epoch ends."""
        del pl_module  # unused
        # If max_epochs is not available, set progress to 0.5
        if trainer.state.stage is not None:
            progress = (
                (trainer.current_epoch + 1) / trainer.max_epochs
                if (trainer.max_epochs is not None and trainer.max_epochs > 0)
                else 0.5
            )
            self._send_progress(progress, trainer.state.stage.value)
        self._check_cancel_training(trainer)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when testing ends."""
        del pl_module  # unused
        if trainer.state.stage is not None:
            self._send_progress(1.0, trainer.state.stage.value)
        self._check_cancel_training(trainer)

    # Predict callbacks
    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when prediction starts."""
        del pl_module  # unused
        if trainer.state.stage is not None:
            self._send_progress(0, trainer.state.stage.value)
        else:
            self._send_progress(0, "Prediction started")
        self._check_cancel_training(trainer)

    def on_predict_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when a prediction batch starts."""
        del pl_module, batch, batch_idx, dataloader_idx  # unused
        self._check_cancel_training(trainer)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when a prediction epoch ends."""
        del pl_module  # unused
        # If max_epochs is not available, set progress to 0.5
        if trainer.state.stage is not None:
            progress = (
                (trainer.current_epoch + 1) / trainer.max_epochs
                if (trainer.max_epochs is not None and trainer.max_epochs > 0)
                else 0.5
            )
            self._send_progress(progress, trainer.state.stage.value)
        self._check_cancel_training(trainer)

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when prediction ends."""
        del pl_module  # unused
        if trainer.state.stage is not None:
            self._send_progress(1.0, trainer.state.stage.value)
        self._check_cancel_training(trainer)
