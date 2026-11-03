# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for temporarily unavailable MVTec downloads."""

from pathlib import Path

import pytest

from anomalib.data import MVTec3D, MVTecAD, MVTecAD2, MVTecLOCO
from anomalib.data.datamodules.base.image import AnomalibDataModule


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if dataset preparation attempts a network connection."""

    def fail_connect(*_args: object, **_kwargs: object) -> None:
        pytest.fail("MVTec dataset preparation must not access the network")

    monkeypatch.setattr("socket.socket.connect", fail_connect)


@pytest.mark.parametrize("datamodule_class", [MVTecAD, MVTecAD2, MVTec3D, MVTecLOCO])
@pytest.mark.parametrize("root_exists", [False, True])
def test_missing_dataset_requires_manual_download(
    datamodule_class: type[AnomalibDataModule],
    root_exists: bool,
    tmp_path: Path,
) -> None:
    """Missing categories fail early with actionable guidance and no downloads."""
    root = tmp_path / "dataset"
    if root_exists:
        (root / "another_category").mkdir(parents=True)
    datamodule = datamodule_class(root=root, category="missing")

    with pytest.raises(FileNotFoundError, match="automatic download") as exc_info:
        datamodule.prepare_data()

    message = str(exc_info.value)
    assert "https://www.mvtec.com/research-teaching/datasets/" in message
    assert str(root / "missing") in message
    assert "https://github.com/open-edge-platform/anomalib/issues/3892" in message
    assert not (root / "missing").exists()
    assert root.exists() == root_exists


@pytest.mark.parametrize("datamodule_class", [MVTecAD, MVTecAD2, MVTec3D, MVTecLOCO])
def test_local_dataset_does_not_download(
    datamodule_class: type[AnomalibDataModule],
    tmp_path: Path,
) -> None:
    """Existing category directories remain usable without network access."""
    category_path = tmp_path / "local"
    category_path.mkdir()
    datamodule = datamodule_class(root=tmp_path, category="local")

    datamodule.prepare_data()

    assert category_path.is_dir()
