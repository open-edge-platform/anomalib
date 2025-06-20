# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PyTorch inferencer for running inference with trained anomaly detection models.

This module provides the PyTorch inferencer implementation for running inference
with trained PyTorch models.

.. warning::
    This is a legacy inferencer. It is recommended to use :class:`anomalib.engine.Engine.predict()`
    instead, which provides a more modern and feature-rich interface for model inference.

.. danger::
    **Security Notice**: PyTorch model loading uses Python's pickle module, which can execute code from the checkpoint
    file.This is a standard PyTorch behavior, not specific to this library. For security, load models only from trusted
    sources and consider using safer formats like ONNX or TorchScript for production use. To proceed with loading, set:

    - Load models only from trusted sources
    - Consider using safer formats like ONNX or TorchScript for production use

    To proceed with loading, set:

    .. code-block:: bash

        export TRUST_REMOTE_CODE=1

Example:
    Assume we have a PyTorch model saved as a ``.pt`` file:

    >>> from anomalib.deploy import TorchInferencer
    >>> model = TorchInferencer(path="path/to/model.pt", device="cpu")

    Make predictions:

    >>> # From image path
    >>> prediction = model.predict("path/to/image.jpg")

    >>> # From PIL Image
    >>> from PIL import Image
    >>> image = Image.open("path/to/image.jpg")
    >>> prediction = model.predict(image)

    >>> # From torch tensor
    >>> import torch
    >>> image = torch.rand(3, 224, 224)
    >>> prediction = model.predict(image)

    The prediction result contains anomaly maps and scores:

    >>> prediction.anomaly_map  # doctest: +SKIP
    tensor([[0.1, 0.2, ...]])

    >>> prediction.pred_score  # doctest: +SKIP
    tensor(0.86)
"""

import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL.Image import Image as PILImage
from torch import nn
from torchvision.transforms.v2.functional import to_dtype, to_image

from anomalib.data import ImageBatch
from anomalib.data.utils import read_image

logger = logging.getLogger(__name__)


class TorchInferencer:
    """PyTorch inferencer for anomaly detection models.

    .. warning::
        This is a legacy inferencer. It is recommended to use :class:`anomalib.engine.Engine.predict()`
        instead, which provides a more modern and feature-rich interface for model inference.

    .. danger::
        **Security Notice**: PyTorch model loading uses Python's pickle module,
        which can execute code from the checkpoint file. This is a standard PyTorch behavior,
        not specific to this library. For security:

        - Load models only from trusted sources
        - Consider using safer formats like ONNX or TorchScript for production use

        To proceed with loading, set:

        .. code-block:: bash

            export TRUST_REMOTE_CODE=1

    Args:
        path (str | Path): Path to the PyTorch model weights file.
        device (str, optional): Device to use for inference.
            Options are ``"auto"``, ``"cpu"``, ``"cuda"``, ``"gpu"``.
            Defaults to ``"auto"``.

    Example:
        >>> from anomalib.deploy import TorchInferencer
        >>> model = TorchInferencer(path="path/to/model.pt")
        >>> predictions = model.predict(image="path/to/image.jpg")

    Raises:
        ValueError: If an invalid device is specified.
        ValueError: If the model file has an unknown extension.
        ValueError: If TRUST_REMOTE_CODE environment variable is not set.
        KeyError: If the checkpoint file does not contain a model.
    """

    def __init__(
        self,
        path: str | Path,
        device: str = "auto",
    ) -> None:
        logger.warning(
            "TorchInferencer is a legacy inferencer. Consider using Engine.predict() instead, "
            "which provides a more modern and feature-rich interface for model inference.",
        )
        self.device = self._get_device(device)

        # Load the model weights and metadata
        self.model = self.load_model(path)

    @staticmethod
    def _get_device(device: str) -> torch.device:
        """Get the device to use for inference.

        Args:
            device (str): Device to use for inference.
                Options are ``"auto"``, ``"cpu"``, ``"cuda"``, ``"gpu"``.

        Returns:
            torch.device: PyTorch device object.

        Raises:
            ValueError: If an invalid device is specified.

        Example:
            >>> model = TorchInferencer(path="path/to/model.pt", device="cpu")
            >>> model.device
            device(type='cpu')
        """
        if device not in {"auto", "cpu", "cuda", "gpu"}:
            msg = f"Unknown device {device}"
            raise ValueError(msg)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "gpu":
            device = "cuda"
        return torch.device(device)

    def _load_checkpoint(self, path: str | Path) -> dict:
        """Load the model checkpoint.

        Args:
            path (str | Path): Path to the PyTorch checkpoint file.

        Returns:
            dict: Dictionary containing the model and metadata.

        Raises:
            ValueError: If the model file has an unknown extension.
            ValueError: If TRUST_REMOTE_CODE environment variable is not set.

        Example:
            >>> model = TorchInferencer(path="path/to/model.pt")
            >>> checkpoint = model._load_checkpoint("path/to/model.pt")
            >>> isinstance(checkpoint, dict)
            True
        """
        if isinstance(path, str):
            path = Path(path)

        if path.suffix not in {".pt", ".pth"}:
            msg = f"Unknown PyTorch checkpoint format {path.suffix}. Make sure you save the PyTorch model."
            raise ValueError(msg)

        trust_remote_code_enabled = os.environ.get("TRUST_REMOTE_CODE", "0").lower() in {"1", "true"}

        if not trust_remote_code_enabled:
            msg = (
                "Loading this model checkpoint requires executing arbitrary code via Python's pickle module, "
                "which is disabled by default for security reasons. This can be exploited by malicious model files. "
                "If you trust the source of this model and understand the risks, "
                "set the environment variable `TRUST_REMOTE_CODE=1` to allow loading."
            )
            raise ValueError(msg)

        logger.warning(
            "TRUST_REMOTE_CODE is set to True. Loading model using pickle module, "
            "which is inherently insecure and can lead to arbitrary code execution. "
            "Only set this to True if you TRUST the source of the checkpoint.",
        )
        # See mitigation details in https://github.com/open-edge-platform/anomalib/pull/2729
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        return torch.load(path, map_location=self.device, weights_only=False)  # nosec B614

    def load_model(self, path: str | Path) -> nn.Module:
        """Load the PyTorch model.

        Args:
            path (str | Path): Path to the PyTorch model file.

        Returns:
            nn.Module: Loaded PyTorch model in evaluation mode.

        Raises:
            KeyError: If the checkpoint file does not contain a model.

        Example:
            >>> model = TorchInferencer(path="path/to/model.pt")
            >>> isinstance(model.model, nn.Module)
            True
        """
        checkpoint = self._load_checkpoint(path)
        if "model" not in checkpoint:
            msg = "``model`` not found in checkpoint. Please check the checkpoint file."
            raise KeyError(msg)

        model = checkpoint["model"]
        model.eval()
        return model.to(self.device)

    def predict(self, image: str | Path | np.ndarray | PILImage | torch.Tensor) -> ImageBatch:
        """Predict anomalies for an input image.

        Args:
            image (str | Path | np.ndarray | PILImage | torch.Tensor): Input image to predict.
                Can be a file path or PyTorch tensor.

        Returns:
            ImageBatch: Prediction results containing anomaly maps and scores.

        Example:
            >>> model = TorchInferencer(path="path/to/model.pt")
            >>> predictions = model.predict("path/to/image.jpg")
            >>> predictions.anomaly_map.shape  # doctest: +SKIP
            torch.Size([1, 256, 256])
        """
        if isinstance(image, str | Path):
            image = read_image(image, as_tensor=True)
        elif isinstance(image, np.ndarray | PILImage):
            image = to_dtype(to_image(image), torch.float32, scale=True)

        image = self.pre_process(image)
        predictions = self.model(image)

        return ImageBatch(image=image, **predictions._asdict())

    def pre_process(self, image: torch.Tensor) -> torch.Tensor:
        """Pre-process the input image.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Pre-processed image tensor.

        Example:
            >>> model = TorchInferencer(path="path/to/model.pt")
            >>> image = torch.rand(3, 224, 224)
            >>> processed = model.pre_process(image)
            >>> processed.shape
            torch.Size([1, 3, 224, 224])
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # model expects [B, C, H, W]

        return image.to(self.device)
