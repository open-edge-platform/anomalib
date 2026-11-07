# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PatchFlow Algorithm Implementation.

PatchFlow is a normalizing-flow-based anomaly detection model that fuses
multi-scale backbone features into a single tensor and runs a single
normalizing flow for detection and localization.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Patchflow
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = Patchflow()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP
    >>> predictions = engine.predict(model, datamodule=datamodule)  # doctest: +SKIP
"""

from .lightning_model import Patchflow
from .loss import PatchflowLoss
from .torch_model import PatchflowModel

__all__ = ["Patchflow", "PatchflowLoss", "PatchflowModel"]
