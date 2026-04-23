# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Max-steps progress callback for step-based training.

Lightning internally sets ``max_epochs=-1`` when only ``max_steps`` is provided.
The Rich progress bar then displays "Epoch X/-2" because it computes
``max_epochs - 1``. This callback estimates the total number of epochs from
``max_steps`` and ``num_training_batches`` so the progress bar shows a
meaningful "Epoch X/N" instead, where *N* is the last epoch index
(``estimated_epochs - 1``, zero-based).

Example:
    Add the callback when training with ``max_steps``::

        from anomalib.callbacks import MaxStepsProgressCallback
        from lightning.pytorch import Trainer

        trainer = Trainer(max_steps=5000, callbacks=[MaxStepsProgressCallback()])

Note:
    This relies on Lightning's ``RichProgressBar._get_train_description``
    as implemented in Lightning 2.x. If the internals change in future
    versions, this callback may need updating.
"""

import math

from lightning.pytorch import Callback, LightningModule, Trainer


class MaxStepsProgressCallback(Callback):
    """Correct epoch display in the Rich progress bar for step-based training.

    When Lightning is configured with ``max_steps`` (and no explicit
    ``max_epochs``), it internally sets ``max_epochs = -1``.  The default
    ``RichProgressBar`` then shows *"Epoch X/-2"* which is confusing.  This
    callback patches the progress bar so it displays the estimated total
    number of epochs derived from ``max_steps / num_training_batches``.
    """

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: PLR6301
        """Patch the Rich progress bar at the start of training.

        Args:
            trainer: PyTorch Lightning trainer instance.
            pl_module: The current training module (unused).
        """
        del pl_module  # Not used.

        if trainer.max_epochs is None or trainer.max_epochs >= 0:
            return

        try:
            from lightning.pytorch.callbacks import RichProgressBar
        except ImportError:
            try:
                from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar
            except ImportError:
                return

        progress_bar = getattr(trainer, "progress_bar_callback", None)
        if not isinstance(progress_bar, RichProgressBar) or not hasattr(
            progress_bar,
            "_get_train_description",
        ):
            return

        num_batches = trainer.num_training_batches
        max_steps = trainer.max_steps
        if max_steps > 0 and isinstance(num_batches, (int, float)) and num_batches > 0 and math.isfinite(num_batches):
            est_max_epochs = math.ceil(max_steps / num_batches)
        else:
            est_max_epochs = None

        val_desc = getattr(progress_bar, "validation_description", "Validation")

        class _FixedRichProgressBar(RichProgressBar):
            """RichProgressBar subclass with corrected epoch display for step-based training."""

            def _get_train_description(self, current_epoch: int) -> str:  # noqa: PLR6301
                desc = f"Epoch {current_epoch}"
                if est_max_epochs is not None:
                    desc += f"/{est_max_epochs - 1}"
                if len(val_desc) > len(desc):
                    desc = f"{desc:{len(val_desc)}}"
                return desc

        progress_bar.__class__ = _FixedRichProgressBar
