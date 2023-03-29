"""Validation loop."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from typing import List

from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.epoch.evaluation_epoch_loop import EvaluationEpochLoop
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

import anomalib.trainer as core


class AnomalibValidationEpochLoop(EvaluationEpochLoop):
    def __init__(self):
        super().__init__()
        self.trainer: core.AnomalibTrainer

    def _evaluation_step_end(self, *args, **kwargs) -> STEP_OUTPUT | None:
        """Runs ``validation_step_end`` after the end of one validation step."""
        outputs = super()._evaluation_step_end(*args, **kwargs)
        if outputs is not None:
            self.trainer.post_processor.compute_labels(outputs)
            self.trainer.post_processor.update(self.trainer.lightning_module, outputs)
            self.trainer.normalizer.update_metrics(outputs)
        return outputs

    @lru_cache(1)
    def _should_track_batch_outputs_for_epoch_end(self) -> bool:
        """Track batch outputs for epoch end.

        If this is not overridden, the outputs are not collected if the model does not have a ``validation_step_end``
        method. This ensures that the outputs are collected even if the model does not have a ``validation_step_end``
        method.
        """
        return True


class AnomalibValidationLoop(EvaluationLoop):
    def __init__(self) -> None:
        super().__init__()
        self.trainer: core.AnomalibTrainer

    def on_run_start(self, *args, **kwargs) -> None:
        self.replace(epoch_loop=AnomalibValidationEpochLoop)
        return super().on_run_start(*args, **kwargs)

    def _evaluation_epoch_end(self, outputs: List[EPOCH_OUTPUT]):
        """Runs ``validation_epoch_end``

        Args:
            outputs (List[EPOCH_OUTPUT]): Outputs
        """
        # with a single dataloader don't pass a 2D list | Taken from base method
        output_or_outputs: EPOCH_OUTPUT | List[EPOCH_OUTPUT] = (
            outputs[0] if len(outputs) > 0 and self.num_dataloaders == 1 else outputs
        )
        self.trainer.post_processor.compute(self.trainer.lightning_module, output_or_outputs)
        self.trainer.metrics_manager.set_threshold(
            self.trainer.lightning_module.image_threshold.value.cpu(),
            self.trainer.lightning_module.pixel_threshold.value.cpu(),
        )
        self.trainer.metrics_manager.compute(output_or_outputs)
        self.trainer.metrics_manager.log(self.trainer, "validation_epoch_end")
        super()._evaluation_epoch_end(outputs)
