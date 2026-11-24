# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for MVTec dataset download behavior."""

from pathlib import Path

import pytest

from anomalib.data import MVTec3D, MVTecAD, MVTecAD2, MVTecLOCO
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datamodules.depth import mvtec_3d
from anomalib.data.datamodules.image import mvtec_loco, mvtecad, mvtecad2
from anomalib.data.datasets.image import mvtec_loco as mvtec_loco_dataset


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if dataset preparation attempts a network connection."""

    def fail_connect(*_args: object, **_kwargs: object) -> None:
        pytest.fail("MVTec dataset preparation must not access the network")

    monkeypatch.setattr("socket.socket.connect", fail_connect)


@pytest.mark.parametrize(
    ("datamodule_class", "module", "expected_info"),
    [
        (MVTecAD, mvtecad, mvtecad.DOWNLOAD_INFO),
        (MVTecAD2, mvtecad2, mvtecad2.DOWNLOAD_INFO),
        (MVTec3D, mvtec_3d, mvtec_3d.DOWNLOAD_INFO),
        (MVTecLOCO, mvtec_loco, mvtec_loco_dataset.DOWNLOAD_INFO),
    ],
)
def test_missing_dataset_triggers_autodownload(
    datamodule_class: type[AnomalibDataModule],
    module: object,
    expected_info: object,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing categories invoke download_and_extract with the module DOWNLOAD_INFO."""
    calls: list[tuple[Path, object]] = []

    def fake_download(root: Path, info: object) -> None:
        calls.append((root, info))

    monkeypatch.setattr(module, "download_and_extract", fake_download)
    root = tmp_path / "dataset"
    datamodule = datamodule_class(root=root, category="missing")

    datamodule.prepare_data()

    assert calls == [(root, expected_info)]


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
