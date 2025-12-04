# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Loading pre-trained DINOv2 Vision Transformer models.

This module provides the :class:`DinoV2Loader` class for constructing and loading
pre-trained DINOv2 Vision Transformer models used in the Dinomaly anomaly detection
framework. It supports both standard DINOv2 models and register-token variants, and
allows custom Vision Transformer factories to be supplied.

Example:
    >>> from anomalib.models.components.dinov2 import DinoV2Loader
    >>> loader = DinoV2Loader()
    >>> model = loader.load("dinov2_vit_base_14")
    >>> model = loader.load("vit_base_14")
    >>> custom_loader = DinoV2Loader(vit_factory=my_custom_vit_module)
    >>> model = custom_loader.load("dinov2reg_vit_base_14")

The DINOv2 loader handles:

- Parsing model names and validating architecture types
- Constructing the appropriate Vision Transformer model
- Locating or downloading the corresponding pre-trained weights
- Supporting custom ViT implementations via a pluggable factory

This enables a simple, unified interface for accessing DINOv2-based backbones in
downstream anomaly detection tasks.
"""

import logging
from pathlib import Path
from typing import ClassVar
from urllib.request import urlretrieve

import torch

from anomalib.data.utils import DownloadInfo
from anomalib.data.utils.download import DownloadProgressBar
from anomalib.models.components.dinov2 import vision_transformer as dinov2_models

logger = logging.getLogger(__name__)

MODEL_FACTORIES: dict[str, object] = {
    "dinov2": dinov2_models,
    "dinov2_reg": dinov2_models,
}


class DinoV2Loader:
    """Simple loader for DINOv2 Vision Transformer models.

    Supports loading dinov2, dinov2_reg, and dinomaly model variants across small, base,
    and large architectures.
    """

    DINOV2_BASE_URL: ClassVar[str] = "https://dl.fbaipublicfiles.com/dinov2"

    MODEL_CONFIGS: ClassVar[dict[str, dict[str, int]]] = {
        "small": {"embed_dim": 384, "num_heads": 6},
        "base": {"embed_dim": 768, "num_heads": 12},
        "large": {"embed_dim": 1024, "num_heads": 16},
    }

    def __init__(
        self,
        cache_dir: str | Path = "./pre_trained/",
        vit_factory: object | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.vit_factory = vit_factory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, model_name: str) -> torch.nn.Module:
        """Load a DINOv2 model by name.

        Args:
            model_name: Model identifier such as "dinov2_vit_base_14".

        Returns:
            A fully constructed and weight-loaded PyTorch module.

        Raises:
            ValueError: If the requested model name is malformed or unsupported.
        """
        model_type, architecture, patch_size = self._parse_name(model_name)
        model = self.create_model(model_type, architecture, patch_size)
        self._load_weights(model, model_type, architecture, patch_size)

        logger.info(f"Loaded model: {model_name}")
        return model

    @classmethod
    def from_name(
        cls,
        model_name: str,
        cache_dir: str | Path = "./pre_trained/",
    ) -> torch.nn.Module:
        """Instantiate a loader and return the requested model."""
        loader = cls(cache_dir)
        return loader.load(model_name)

    def _parse_name(self, name: str) -> tuple[str, str, int]:
        """Parse a model name string into components.

        Args:
            name: Full model name string.

        Returns:
            Tuple of (model_type, architecture_name, patch_size).

        Raises:
            ValueError: If the prefix or architecture is unknown.
        """
        parts = name.split("_")
        prefix = parts[0]
        architecture = parts[-2]
        patch_size = int(parts[-1])

        if prefix == "dinov2reg":
            model_type = "dinov2_reg"
        elif prefix == "dinov2":
            model_type = "dinov2"
        elif prefix == "dinomaly":
            model_type = "dinomaly"
        else:
            msg = f"Unknown model type prefix '{prefix}'."
            raise ValueError(msg)

        if architecture not in self.MODEL_CONFIGS:
            msg = f"Invalid architecture '{architecture}'. Expected one of: {list(self.MODEL_CONFIGS)}"
            raise ValueError(
                msg,
            )

        return model_type, architecture, patch_size

    def create_model(self, model_type: str, architecture: str, patch_size: int) -> torch.nn.Module:
        """Create a Vision Transformer model.

        Args:
            model_type: Normalized model family name (e.g., "dinov2", "dinov2_reg").
            architecture: Architecture size (e.g., "small", "base", "large").
            patch_size: ViT patch size.

        Returns:
            Instantiated Vision Transformer model.

        Raises:
            ValueError: If no matching constructor exists.
        """
        model_kwargs = {
            "patch_size": patch_size,
            "img_size": 518,
            "block_chunks": 0,
            "init_values": 1e-8,
            "interpolate_antialias": False,
            "interpolate_offset": 0.1,
        }

        if model_type == "dinov2_reg":
            model_kwargs["num_register_tokens"] = 4

        # If user supplied a custom ViT module, use it
        module = self.vit_factory if self.vit_factory is not None else MODEL_FACTORIES[model_type]

        ctor = getattr(module, f"vit_{architecture}", None)
        if ctor is None:
            msg = f"No constructor vit_{architecture} in module {module}"
            raise ValueError(msg)

        return ctor(**model_kwargs)

    def _load_weights(
        self,
        model: torch.nn.Module,
        model_type: str,
        architecture: str,
        patch_size: int,
    ) -> None:
        """Load pre-trained weights from disk, downloading them if necessary."""
        weight_path = self._get_weight_path(model_type, architecture, patch_size)

        if not weight_path.exists():
            self._download_weights(model_type, architecture, patch_size)

        # Weights_only is set to True
        # See mitigation details in https://github.com/open-edge-platform/anomalib/pull/2729
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)  # nosec B614
        model.load_state_dict(state_dict, strict=False)

    def _get_weight_path(
        self,
        model_type: str,
        architecture: str,
        patch_size: int,
    ) -> Path:
        """Return the expected local path for downloaded weights."""
        arch_code = architecture[0]

        if model_type == "dinov2_reg":
            filename = f"dinov2_vit{arch_code}{patch_size}_reg4_pretrain.pth"
        else:
            filename = f"dinov2_vit{arch_code}{patch_size}_pretrain.pth"

        return self.cache_dir / filename

    def _download_weights(
        self,
        model_type: str,
        architecture: str,
        patch_size: int,
    ) -> None:
        """Download DINOv2 weight files using Anomalib's standardized utilities."""
        weight_path = self._get_weight_path(model_type, architecture, patch_size)
        arch_code = architecture[0]

        model_dir = f"dinov2_vit{arch_code}{patch_size}"
        url = f"{self.DINOV2_BASE_URL}/{model_dir}/{weight_path.name}"

        download_info = DownloadInfo(
            name=f"DINOv2 {model_type} {architecture} weights",
            url=url,
            hashsum="",  # DINOv2 publishes no official hash
            filename=weight_path.name,
        )

        logger.info(
            f"Downloading DINOv2 weights: {weight_path.name} to {self.cache_dir}",
        )

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        with DownloadProgressBar(
            unit="B",
            unit_scale=True,
            miniters=1,
            desc=download_info.name,
        ) as progress_bar:
            # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected  # noqa: ERA001, E501
            urlretrieve(  # noqa: S310  # nosec B310
                url=url,
                filename=weight_path,
                reporthook=progress_bar.update_to,
            )
