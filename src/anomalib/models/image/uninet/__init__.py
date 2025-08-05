# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""UniNet Model for anomaly detection.

This module implements anomaly detection using the UniNet model. It is a model designed for diverse domains and is
suited for both supervised and unsupervised anomaly detection. It also focuses on multi-class anomaly detection.

Example:
    >>> from anomalib.models import UniNet
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.engine import Engine

    >>> # Initialize model and data
    >>> datamodule = MVTecAD()
    >>> model = UniNet()

    >>> # Train the model
    >>> engine = Engine()
    >>> engine.train(model=model, datamodule=datamodule)

    >>> # Get predictions
    >>> engine.predict(model=model, datamodule=datamodule)

See Also:
    - :class:`UniNet`: Main model class for UniNet-based anomaly detection
    - :class:`UniNetModel`: PyTorch implementation of the UniNet model
"""

from .lightning_model import UniNet
from .torch_model import UniNetModel

__all__ = ["UniNet", "UniNetModel"]
