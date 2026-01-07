# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for DinoV2Loader."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from anomalib.models.components.dinov2.dinov2_loader import DinoV2Loader


@pytest.fixture()
def dummy_model() -> nn.Module:
    """Return a simple dummy model used by fake constructors."""

    class Dummy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

    return Dummy()


@pytest.fixture()
def loader() -> DinoV2Loader:
    """Return a loader instance with a non-functional cache path."""
    return DinoV2Loader(cache_dir="not_used_in_unit_tests")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("dinov2_vit_base_14", ("dinov2", "base", 14)),
        ("dinov2reg_vit_small_16", ("dinov2_reg", "small", 16)),
        ("dinomaly_vit_large_14", ("dinomaly", "large", 14)),
    ],
)
def test_parse_name_valid(
    loader: DinoV2Loader,
    name: str,
    expected: tuple[str, str, int],
) -> None:
    """Validate that supported model names parse correctly."""
    assert loader._parse_name(name) == expected  # noqa: SLF001


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("foo_vit_base_14", "foo"),
        ("x_vit_small_16", "x"),
        ("wrongprefix_vit_large_14", "wrongprefix"),
    ],
)
def test_parse_name_invalid_prefix(loader: DinoV2Loader, name: str, expected: str) -> None:
    """Ensure invalid model prefixes raise ValueError."""
    msg = f"Unknown model type prefix '{expected}'."
    with pytest.raises(ValueError, match=msg):
        loader._parse_name(name)  # noqa: SLF001


def test_parse_name_invalid_architecture(loader: DinoV2Loader) -> None:
    """Ensure unknown architecture names raise ValueError."""
    expected_msg = f"Invalid architecture 'tiny'. Expected one of: {list(loader.MODEL_CONFIGS)}"
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        loader._parse_name("dinov2_vit_tiny_14")  # noqa: SLF001


def test_create_model_success(loader: DinoV2Loader, dummy_model: nn.Module) -> None:
    """Verify model creation succeeds when constructor exists."""
    fake_module = MagicMock()
    fake_module.vit_small = MagicMock(return_value=dummy_model)

    loader.vit_factory = fake_module
    model = loader.create_model("dinov2", "small", 14)

    fake_module.vit_small.assert_called_once()
    assert model is dummy_model


def test_create_model_missing_constructor(loader: DinoV2Loader) -> None:
    """Verify missing constructors cause ValueError."""
    loader.vit_factory = object()
    expected_msg = f"No constructor vit_base in module {loader.vit_factory}"
    with pytest.raises(ValueError, match=expected_msg):
        loader.create_model("dinov2", "base", 14)


def test_get_weight_path_dinov2(loader: DinoV2Loader) -> None:
    """Check generated weight filename for default dinov2 models."""
    path = loader._get_weight_path("dinov2", "base", 14)  # noqa: SLF001
    assert path.name == "dinov2_vitb14_pretrain.pth"


def test_get_weight_path_reg(loader: DinoV2Loader) -> None:
    """Check generated weight filename for register-token models."""
    path = loader._get_weight_path("dinov2_reg", "large", 16)  # noqa: SLF001
    assert path.name == "dinov2_vitl16_reg4_pretrain.pth"


@patch("anomalib.models.components.dinov2.dinov2_loader.torch.load")
@patch("anomalib.models.components.dinov2.dinov2_loader.DinoV2Loader._download_weights")
def test_load_calls_weight_loading(
    mock_download: MagicMock,
    mock_torch_load: MagicMock,
    loader: DinoV2Loader,
    dummy_model: nn.Module,
) -> None:
    """Confirm load() uses existing weights without downloading."""
    fake_module = MagicMock()
    fake_module.vit_base = MagicMock(return_value=dummy_model)
    loader.vit_factory = fake_module

    fake_path = MagicMock()
    fake_path.exists.return_value = True
    loader._get_weight_path = MagicMock(return_value=fake_path)  # noqa: SLF001

    mock_torch_load.return_value = {"layer": torch.zeros(1)}

    loaded = loader.load("dinov2_vit_base_14")

    fake_module.vit_base.assert_called_once()
    mock_download.assert_not_called()
    mock_torch_load.assert_called_once()
    assert loaded is dummy_model


@patch("anomalib.models.components.dinov2.dinov2_loader.torch.load")
@patch("anomalib.models.components.dinov2.dinov2_loader.DinoV2Loader._download_weights")
def test_load_triggers_download_when_missing(
    mock_download: MagicMock,
    mock_torch_load: MagicMock,
    loader: DinoV2Loader,
    dummy_model: nn.Module,
) -> None:
    """Confirm load() triggers weight download when file is missing."""
    fake_module = MagicMock()
    fake_module.vit_small = MagicMock(return_value=dummy_model)
    loader.vit_factory = fake_module

    fake_path = MagicMock()
    fake_path.exists.return_value = False
    loader._get_weight_path = MagicMock(return_value=fake_path)  # noqa: SLF001

    mock_torch_load.return_value = {"test": torch.zeros(1)}

    loader.load("dinov2_vit_small_14")

    mock_download.assert_called_once()
    mock_torch_load.assert_called_once()
    fake_module.vit_small.assert_called_once()
