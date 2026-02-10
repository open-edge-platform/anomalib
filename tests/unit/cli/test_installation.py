# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for installation utils."""

import os
import platform
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from pkg_resources import Requirement
import anomalib.cli.utils.installation as installation

from anomalib.cli.utils.installation import (
    get_cuda_suffix,
    get_cuda_version,
    get_hardware_suffix,
    get_requirements,
    get_torch_install_args,
    parse_requirements,
    update_cuda_version_with_available_torch_cuda_build,
)


@pytest.fixture()
def requirements_file() -> Path:
    """Create a temporary requirements file with some example requirements."""
    requirements = ["numpy==1.19.5", "opencv-python-headless>=4.5.1.48"]
    with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
        f.write("\n".join(requirements))
        return Path(f.name)


def test_get_requirements(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_requirements returns the expected dictionary of requirements."""
    requirements = get_requirements("anomalib")
    assert isinstance(requirements, dict)
    assert len(requirements) > 0
    for reqs in requirements.values():
        assert isinstance(reqs, list)
        for req in reqs:
            assert isinstance(req, Requirement)
    monkeypatch.setattr(installation, "requires", lambda _module="anomalib": None)
    assert get_requirements() == {}


def test_parse_requirements() -> None:
    """Test that parse_requirements returns the expected tuple of requirements."""
    requirements = [
        Requirement.parse("torch==2.0.0"),
        Requirement.parse("onnx>=1.8.1"),
    ]
    torch_req, other_reqs = parse_requirements(requirements)
    assert isinstance(torch_req, str)
    assert isinstance(other_reqs, list)
    assert torch_req == "torch==2.0.0"
    assert other_reqs == ["onnx>=1.8.1"]

    requirements = [
        Requirement.parse("torch<=2.0.1, >=1.8.1"),
    ]
    torch_req, other_reqs = parse_requirements(requirements)
    assert torch_req == "torch<=2.0.1,>=1.8.1"
    assert other_reqs == []

    requirements = [
        Requirement.parse("onnx>=1.8.1"),
    ]
    with pytest.raises(ValueError, match=r"Could not find torch requirement."):
        parse_requirements(requirements)


def test_get_cuda_version_with_version_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that get_cuda_version returns the expected CUDA version when version file exists."""
    tmp_path = tmp_path / "cuda"
    tmp_path.mkdir()
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    version_file = tmp_path / "version.json"
    version_file.write_text('{"cuda": {"version": "11.2.0"}}')
    assert get_cuda_version() == "11.2"


def test_get_cuda_version_with_nvcc(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_cuda_version returns the expected CUDA version when nvcc is available."""
    monkeypatch.setattr(Path, "exists", lambda *_args, **_kwargs: False)
    popen_mock = lambda *_args, **_kwargs: SimpleNamespace(
        read=lambda: "Build cuda_11.2.r11.2/compiler.00000_0",
    )
    monkeypatch.setattr(os, "popen", popen_mock)
    assert get_cuda_version() == "11.2"

    def raise_file_not_found(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(os, "popen", raise_file_not_found)
    assert get_cuda_version() is None


def test_update_cuda_version_with_available_torch_cuda_build() -> None:
    """Test that update_cuda_version_with_available_torch_cuda_build returns the expected CUDA version."""
    assert update_cuda_version_with_available_torch_cuda_build("11.1", "2.0.1") == "11.7"
    assert update_cuda_version_with_available_torch_cuda_build("11.7", "2.0.1") == "11.7"
    assert update_cuda_version_with_available_torch_cuda_build("11.8", "2.0.1") == "11.8"
    assert update_cuda_version_with_available_torch_cuda_build("12.1", "2.1.1") == "12.1"


def test_get_cuda_suffix() -> None:
    """Test the get_cuda_suffix function."""
    assert get_cuda_suffix(cuda_version="11.2") == "cu112"
    assert get_cuda_suffix(cuda_version="11.8") == "cu118"


def test_get_hardware_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the behavior of the get_hardware_suffix function."""
    monkeypatch.setattr(installation, "get_cuda_version", lambda: "11.2")
    assert get_hardware_suffix() == "cu112"

    monkeypatch.setattr(installation, "get_cuda_version", lambda: "12.1")
    assert get_hardware_suffix(with_available_torch_build=True, torch_version="2.0.1") == "cu118"

    with pytest.raises(ValueError, match=r"``torch_version`` must be provided"):
        get_hardware_suffix(with_available_torch_build=True)

    monkeypatch.setattr(installation, "get_cuda_version", lambda: None)
    assert get_hardware_suffix() == "cpu"


def test_get_torch_install_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_torch_install_args returns the expected install arguments."""
    requirement = Requirement.parse("torch>=2.1.1")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(installation, "get_hardware_suffix", lambda *_args, **_kwargs: "cpu")
    install_args = get_torch_install_args(requirement)
    expected_args = [
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
        "torch>=2.1.1",
        "torchvision>=0.16.1",
    ]
    for arg in expected_args:
        assert arg in install_args

    requirement = Requirement.parse("torch>=1.13.0,<=2.0.1")
    monkeypatch.setattr(installation, "get_hardware_suffix", lambda *_args, **_kwargs: "cu111")
    install_args = get_torch_install_args(requirement)
    expected_args = [
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu111",
    ]
    for arg in expected_args:
        assert arg in install_args

    requirement = Requirement.parse("torch==2.0.1")
    expected_args = [
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu111",
        "torch==2.0.1",
        "torchvision==0.15.2",
    ]
    install_args = get_torch_install_args(requirement)
    for arg in expected_args:
        assert arg in install_args

    install_args = get_torch_install_args("torch")
    assert install_args == ["torch"]

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    requirement = Requirement.parse("torch==2.0.1")
    install_args = get_torch_install_args(requirement)
    assert install_args == ["torch==2.0.1"]

    monkeypatch.setattr(platform, "system", lambda: "Unknown")
    with pytest.raises(RuntimeError, match=r"Unsupported OS: Unknown"):
        get_torch_install_args(requirement)
