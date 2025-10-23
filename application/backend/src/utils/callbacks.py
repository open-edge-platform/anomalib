# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightning callback for sending progress to the frontend via the Plugin API."""

from __future__ import annotations

import logging
from ctypes import c_char_p
from multiprocessing import Event, Value
from typing import TYPE_CHECKING, Any

from lightning.pytorch.callbacks import Callback

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer
    from lightning.pytorch.trainer.states import RunningStage

logger = logging.getLogger(__name__)


class ProgressSyncParams:
    def __init__(self) -> None:
        self.progress = Value("f", 0.0)
        self.stage = Value(c_char_p, b"idle")
        self.cancel_training_event = Event()

    def set_stage(self, stage: str) -> None:
        with self.stage.get_lock():
            self.stage.value = stage.encode("utf-8")
        logger.info("Set stage: %s", stage)

    def get_stage(self) -> str:
        return self.stage.value.decode("utf-8")

    def set_progress(self, progress: float) -> None:
        with self.progress.get_lock():
            self.progress.value = progress
        logger.info("Set progress: %s", progress)

    def get_progress(self) -> float:
        return self.progress.value

    def set_cancel_training_event(self) -> None:
        self.cancel_training_event.set()
        logger.info("Set cancel training event")


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

    def _send_progress(self, progress: float, stage: RunningStage) -> None:
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
            logger.info("Sent progress: %s - %d%%", stage.name, progress_percent)
            self.synchronization_parameters.set_progress(progress_percent)
            self.synchronization_parameters.set_stage(stage.name)
        except Exception as e:
            logger.warning("Failed to send progress to event queue: %s", e)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self._send_progress(0, stage)

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: RunningStage) -> None:
        self._send_progress(1.0, stage)

    # Training callbacks
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training starts."""
        self._send_progress(0, trainer.state.stage)
        self._check_cancel_training(trainer)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        """Called when a training batch starts."""
        self._check_cancel_training(trainer)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when a training epoch ends."""
        progress = (trainer.current_epoch + 1) / trainer.max_epochs
        self._send_progress(progress, trainer.state.stage)
        self._check_cancel_training(trainer)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training ends."""
        self._send_progress(1.0, trainer.state.stage)
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
        self._send_progress(0, trainer.state.stage)
        self._check_cancel_training(trainer)

    def on_test_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Called when a test batch starts."""
        self._check_cancel_training(trainer)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when a test epoch ends."""
        progress = (trainer.current_epoch + 1) / trainer.max_epochs if trainer.max_epochs else 0.5
        self._send_progress(progress, trainer.state.stage)
        self._check_cancel_training(trainer)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when testing ends."""
        self._send_progress(1.0, trainer.state.stage)
        self._check_cancel_training(trainer)

    # Predict callbacks
    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when prediction starts."""
        self._send_progress(0, trainer.state.stage)
        self._check_cancel_training(trainer)

    def on_predict_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Called when a prediction batch starts."""
        self._check_cancel_training(trainer)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when a prediction epoch ends."""
        progress = (trainer.current_epoch + 1) / trainer.max_epochs if trainer.max_epochs else 0.5
        self._send_progress(progress, trainer.state.stage)
        self._check_cancel_training(trainer)

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when prediction ends."""
        self._send_progress(1.0, trainer.state.stage)
        self._check_cancel_training(trainer)
