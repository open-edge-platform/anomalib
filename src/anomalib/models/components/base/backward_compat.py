# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatibility helpers for loading older anomalib checkpoints.

Several DINOv2-based models replaced their custom Vision Transformer encoder (previously
loaded via the now-removed ``DinoV2Loader``) with a frozen
:class:`~anomalib.models.components.feature_extractors.TimmFeatureExtractor`. Checkpoints
trained before that migration store the encoder weights under a different key layout, so a
strict ``load_state_dict`` would fail. Because the encoder is frozen (its weights come
straight from the pretrained backbone), the legacy encoder weights can simply be dropped and
replaced by the freshly-initialised timm encoder weights of the current model.
"""

import logging
from typing import Any

from torch import nn

from anomalib.utils import deprecate

logger = logging.getLogger(__name__)


@deprecate(
    since="2.5.0",
    remove="2.7.0",
    reason="Loading Vision Transformer based checkpoints trained before the "
    "timm-encoder migration is no longer supported",
)
def restore_frozen_encoder_weights(module: nn.Module, checkpoint: dict[str, Any], encoder_key: str) -> None:
    """Migrate a legacy frozen-encoder subtree in a checkpoint to the current layout.

    Replaces the encoder weights stored under ``model.<encoder_key>.*`` in
    ``checkpoint["state_dict"]`` with the current module's freshly-initialised encoder weights,
    leaving every other (trainable) tensor untouched. This makes checkpoints trained before the
    timm-encoder migration loadable with the default strict ``load_state_dict``. It is a no-op
    when the encoder subtree already matches the current model, so new checkpoints pass through
    unchanged.

    Because only the ``model.<encoder_key>.`` subtree is touched, unrelated frozen state such as
    memory banks or normalization buffers is preserved.

    Args:
        module (nn.Module): The LightningModule the checkpoint is being loaded into. Its torch
            model is assumed to be the ``model`` attribute (the anomalib convention), so encoder
            keys are prefixed with ``model.<encoder_key>.``.
        checkpoint (dict[str, Any]): Checkpoint dictionary being loaded, modified in place.
        encoder_key (str): Dotted attribute path of the frozen encoder within the torch model,
            e.g. ``"encoder"``, ``"feature_encoder"`` or ``"teacher.fe"``.
    """
    state_dict = checkpoint.get("state_dict")
    if not state_dict:
        return

    prefix = f"model.{encoder_key}."
    current_encoder = {key: value for key, value in module.state_dict().items() if key.startswith(prefix)}
    legacy_encoder_keys = [key for key in state_dict if key.startswith(prefix)]

    if set(legacy_encoder_keys) == set(current_encoder):
        # Encoder subtree already matches the current model - nothing to migrate.
        return

    logger.info(
        "Migrating legacy frozen encoder '%s' (%d checkpoint tensors -> %d current tensors) "
        "for backward compatibility.",
        encoder_key,
        len(legacy_encoder_keys),
        len(current_encoder),
    )

    for key in legacy_encoder_keys:
        del state_dict[key]
    state_dict.update(current_encoder)
